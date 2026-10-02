"""Replay runner: label-free tests of the allocation model (paper, Sec. 4.4).

The run records store, for every round, the pool ids that were queried but not
the margins of the pool. This runner rebuilds each split with the *same*
`prepare_split` and seeds as the campaign, replays every run from its recorded
selections (labelled set at round r = init + everything selected before r),
refits the classifier with the run's model seed, and measures:

  * q_tau  : fraction of the batch chosen at round r whose margin >= tau
             (i.e. outside the uncertain band U_tau, primary tau = 0.1);
  * q_20   : fraction of the batch outside the 20% lowest-margin pool points;
  * D_hat  : (round 0, once per split x classifier) allocation imbalance of the
             margin batch: the lowest-margin 20% of the pool is clustered into
             M = k regions (k-means), the k margin points are counted per region,
             D_hat = 1/2 ||n_sorted - n_bal_sorted||_1 / k  (fraction of the batch
             that would have to move to balance it). Also with M = k/2.

Exactness is checked, not assumed:
  * pairing_key of the rebuilt split == the record's pairing_key;
  * at every replayed round, band_width_tau0.1 recomputed == recorded value;
  * at round 0, every point of the margin run's recorded batch is among the k
    smallest margins (tie-consistent: random-forest margins are often tied).
Any mismatch is reported in the output (column `ok`) and such rows are
excluded by the analysis.

Usage (from the repository root, single-threaded numerics as in the campaign):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m alframework.bench.replay --records <dir with synthetic/ tabular/ latent/> \
      --data-root data --protocol configs/protocol.json --out replay.parquet \
      [--window n95|full] [--workers 4] [--regimes synthetic tabular latent]

Resumable: tasks already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans

from .data import prepare_split, split_seeds, dataset_n_init
from .run import build_model, _margins

MAIN_PANEL = ["random", "margin", "entropy", "least_confident", "coreset_greedy",
              "coreset_kmeanspp", "typiclust", "probcover", "badge",
              "unc_feature_kmeans", "diversity_opt_batch", "qbc", "robust_qbc",
              "tri_committee", "adaptive_disagreement", "dbal", "rank2022"]
TAU = 0.1
TOL = 1e-9


def _imbalance(region_of_batch: np.ndarray, k: int, M: int) -> float:
    n = np.sort(np.bincount(region_of_batch, minlength=M))[::-1]
    bal = np.full(M, k // M)
    bal[: k - bal.sum()] += 1
    return float(0.5 * np.abs(n - np.sort(bal)[::-1]).sum() / k)


def _n95(records_dir: Path, regime: str, dataset: str, clf: str) -> int:
    """First budget at which the split-averaged margin curve reaches 95% of its max."""
    curves = []
    for f in sorted((records_dir / regime / dataset / clf).glob("split*/margin.json")):
        h = json.load(open(f))["history"]
        curves.append(pd.Series({s["n_labeled"]: s["f1_macro"] for s in h}))
    c = pd.concat(curves, axis=1).mean(axis=1).sort_index()
    first = c[c >= 0.95 * c.max()].index[0]
    return int(first if first > c.index.min() else c.index.max())


def replay_task(task: dict) -> list:
    """All runs of one (regime, dataset, split): both classifiers, all strategies."""
    protocol = task["protocol"]
    regime, dataset, split = task["regime"], task["dataset"], task["split"]
    rcfg = protocol["regimes"][regime]
    rdir = Path(task["records"]) / regime / dataset
    seeds = split_seeds(protocol["seeds"]["base"], protocol["seeds"]["stride"], split)
    sd = prepare_split(regime, dataset, rcfg, task["data_root"], seeds, task.get("cache"))
    k = rcfg["batch_size"]
    rows = []
    for clf in ["lr", "rf"]:
        sdir = rdir / clf / f"split{split:02d}"
        if not sdir.exists():
            continue
        nwin = task["nwin"].get(clf)
        base = build_model(protocol["classifiers"][clf], seeds["model"])
        cache_r0 = {}
        for strat in task["strategies"]:
            fp = sdir / f"{strat}.json"
            if not fp.exists():
                continue
            rec = json.load(open(fp))
            ok_pair = rec["data"]["pairing_key"] == sd.provenance["pairing_key"]
            lab = np.zeros(len(sd.y_pool), bool)
            lab[sd.init_idx] = True
            ids = np.arange(len(sd.y_pool))
            for step in rec["history"]:
                if "selected" not in step:
                    break
                if task["window"] == "n95" and step["n_labeled"] >= nwin:
                    break
                r = step["round"]
                U = ids[~lab]
                if r == 0 and "m" in cache_r0:
                    m = cache_r0["m"]
                else:
                    model = clone(base).fit(sd.X_pool[ids[lab]], sd.y_pool[ids[lab]])
                    m = _margins(model.predict_proba(sd.X_pool[U]))
                    if r == 0:
                        cache_r0["m"] = m
                ok_band = abs(float((m < TAU).mean()) - step.get("band_width_tau0.1", np.nan)) < TOL
                pos = np.searchsorted(U, np.asarray(step["selected"]))
                ms = m[pos]
                thr20 = np.sort(m, kind="stable")[max(1, int(0.2 * len(m))) - 1]
                row = dict(regime=regime, dataset=dataset, classifier=clf, split=split,
                           strategy=strat, round=r, n_labeled=step["n_labeled"],
                           q_tau=float((ms >= TAU).mean()), q_20=float((ms > thr20).mean()),
                           w_tau=float((m < TAU).mean()), ok=bool(ok_pair and ok_band))
                if r == 0 and strat == "margin":
                    order = np.argsort(m, kind="stable")
                    kth = np.sort(m)[k - 1]
                    # tie-consistent: every recorded point is among the k smallest margins
                    ok_sel = bool(np.all(ms <= kth + TOL))
                    n20 = max(2 * k, int(0.2 * len(m)))
                    cand = np.unique(np.concatenate([order[:n20], pos]))
                    where = {int(c): i for i, c in enumerate(cand)}
                    bpos = [where[int(p)] for p in pos]          # the batch actually queried
                    for M, name in [(k, "D_hat"), (max(2, k // 2), "D_hat_half")]:
                        lab_c = KMeans(n_clusters=M, n_init=3,
                                       random_state=seeds["strategy"] % (2**31)).fit_predict(sd.X_pool[U][cand])
                        row[name] = _imbalance(lab_c[bpos], k, M)
                    row["ok"] = bool(row["ok"] and ok_sel)
                    row["ok_margin_selection"] = ok_sel
                rows.append(row)
                lab[np.asarray(step["selected"])] = True
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--protocol", default="configs/protocol.json")
    ap.add_argument("--out", default="replay.parquet")
    ap.add_argument("--cache", default=".cache_txt")
    ap.add_argument("--window", choices=["n95", "full"], default="n95")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--regimes", nargs="*", default=["synthetic", "tabular", "latent"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--splits", type=int, nargs="*", default=None)
    ap.add_argument("--strategies", nargs="*", default=MAIN_PANEL)
    a = ap.parse_args(argv)

    protocol = json.load(open(a.protocol))
    records = Path(a.records)
    out = Path(a.out)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    done_keys = set(map(tuple, done[["regime", "dataset", "split"]].drop_duplicates().values)) if len(done) else set()

    tasks = []
    for regime in a.regimes:
        rc = protocol["regimes"][regime]
        for dataset in rc["datasets"]:
            if a.datasets and dataset not in a.datasets:
                continue
            if not (records / regime / dataset).exists():
                continue
            nwin = {c: _n95(records, regime, dataset, c) for c in ["lr", "rf"]
                    if (records / regime / dataset / c).exists()}
            for s in (a.splits if a.splits is not None else range(rc["splits"])):
                if (regime, dataset, s) in done_keys:
                    continue
                tasks.append(dict(protocol=protocol, regime=regime, dataset=dataset, split=s,
                                  records=str(records), data_root=a.data_root, cache=a.cache,
                                  window=a.window, nwin=nwin, strategies=a.strategies))
    print(f"[replay] {len(tasks)} (dataset, split) tasks, {a.workers} workers", flush=True)
    rows = done.to_dict("records") if len(done) else []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(replay_task, t): (t["regime"], t["dataset"], t["split"]) for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            key = futs[f]
            try:
                r = f.result()
                rows += r
                bad = sum(not x["ok"] for x in r)
                print(f"[replay] {i}/{len(tasks)} {key} rows={len(r)} mismatches={bad} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"[replay] {i}/{len(tasks)} {key} FAILED: {e!r}", flush=True)
            if i % 10 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_parquet(out)
    pd.DataFrame(rows).to_parquet(out)
    print(f"[replay] done -> {out}")


if __name__ == "__main__":
    main()
