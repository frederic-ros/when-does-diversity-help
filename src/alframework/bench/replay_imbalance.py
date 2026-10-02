"""Robustness of the allocation-imbalance proxy D_hat (paper Sec. 4.4, (i)).

Replays the *margin* runs only (same splits, seeds, classifier as the campaign,
selections read from the records) and computes, at round 0 and at up to
`--rounds` rounds evenly spaced in the early-budget window, the imbalance of the
batch actually queried by margin over a partition of the informative region:

  region  : "U01"   = {x in pool : m(x) < 0.1}  (the band of Definition 2), union batch
            "top20" = the 20% lowest-margin pool points, union batch
  M       : k/2, k, 2k clusters (k-means on the region), capped at |region| - 1;
            `degenerate` flags regions too small to be partitioned into M clusters.

D_hat = 1/2 || sort(n) - sort(n_bal) ||_1 / k, with n the counts of the k batch
points per cluster and n_bal the balanced allocation of k points over M clusters.

Usage:
  OMP_NUM_THREADS=1 python -m alframework.bench.replay_imbalance --records <dir> \
      --data-root data --out replay_imbalance.parquet [--rounds 5] [--workers 2]
"""
from __future__ import annotations

import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans

from .data import prepare_split, split_seeds
from .run import build_model, _margins
from .replay import _n95, _imbalance, TAU, TOL


def task_run(t: dict) -> list:
    protocol = t["protocol"]; regime, dataset, split = t["regime"], t["dataset"], t["split"]
    rcfg = protocol["regimes"][regime]
    seeds = split_seeds(protocol["seeds"]["base"], protocol["seeds"]["stride"], split)
    sd = prepare_split(regime, dataset, rcfg, t["data_root"], seeds, t.get("cache"))
    k = rcfg["batch_size"]
    rows = []
    for clf in ["lr", "rf"]:
        fp = Path(t["records"]) / regime / dataset / clf / f"split{split:02d}" / "margin.json"
        if not fp.exists():
            continue
        rec = json.load(open(fp))
        hist = [h for h in rec["history"] if "selected" in h and h["n_labeled"] < t["nwin"][clf]]
        idx = np.unique(np.round(np.linspace(0, len(hist) - 1, min(t["rounds"], len(hist)))).astype(int))
        keep = {hist[i]["round"] for i in idx} | {0}
        base = build_model(protocol["classifiers"][clf], seeds["model"])
        lab = np.zeros(len(sd.y_pool), bool); lab[sd.init_idx] = True
        ids = np.arange(len(sd.y_pool))
        for h in hist:
            r = h["round"]
            if r in keep:
                U = ids[~lab]
                model = clone(base).fit(sd.X_pool[ids[lab]], sd.y_pool[ids[lab]])
                m = _margins(model.predict_proba(sd.X_pool[U]))
                ok = abs(float((m < TAU).mean()) - h.get("band_width_tau0.1", np.nan)) < TOL
                pos = np.searchsorted(U, np.asarray(h["selected"]))
                XU = sd.X_pool[U]
                regions = {
                    "U01": np.unique(np.concatenate([np.flatnonzero(m < TAU), pos])),
                    "top20": np.unique(np.concatenate([np.argsort(m, kind="stable")[:max(2 * k, int(0.2 * len(m)))], pos])),
                }
                for rname, cand in regions.items():
                    where = {int(c): i for i, c in enumerate(cand)}
                    bpos = np.array([where[int(p)] for p in pos])
                    for mname, M in [("half", max(2, k // 2)), ("k", k), ("2k", 2 * k)]:
                        Meff = int(min(M, len(cand) - 1))
                        degenerate = Meff < M
                        lab_c = KMeans(n_clusters=max(2, Meff), n_init=1,
                                       random_state=seeds["strategy"] % (2**31)).fit_predict(XU[cand])
                        rows.append(dict(regime=regime, dataset=dataset, classifier=clf, split=split,
                                         round=r, n_labeled=h["n_labeled"], region=rname, M=mname,
                                         D_hat=_imbalance(lab_c[bpos], k, max(2, Meff)),
                                         region_size=len(cand), degenerate=bool(degenerate),
                                         w_tau=float((m < TAU).mean()), ok=bool(ok)))
            lab[np.asarray(h["selected"])] = True
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--protocol", default="configs/protocol.json")
    ap.add_argument("--out", default="replay_imbalance.parquet")
    ap.add_argument("--cache", default=".cache_txt")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    protocol = json.load(open(a.protocol)); records = Path(a.records); out = Path(a.out)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    have = set(map(tuple, done[["regime", "dataset", "split"]].drop_duplicates().values)) if len(done) else set()
    tasks = []
    for regime, rc in protocol["regimes"].items():
        for dataset in rc["datasets"]:
            if not (records / regime / dataset).exists():
                continue
            nwin = {c: _n95(records, regime, dataset, c) for c in ["lr", "rf"]}
            for s in range(rc["splits"]):
                if (regime, dataset, s) not in have:
                    tasks.append(dict(protocol=protocol, regime=regime, dataset=dataset, split=s,
                                      records=str(records), data_root=a.data_root, cache=a.cache,
                                      nwin=nwin, rounds=a.rounds))
    print(f"[imbalance] {len(tasks)} tasks", flush=True)
    rows = done.to_dict("records") if len(done) else []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(task_run, t): (t["regime"], t["dataset"], t["split"]) for t in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rows += f.result()
            except Exception as e:
                print(f"[imbalance] {futs[f]} FAILED {e!r}", flush=True)
            if i % 20 == 0 or i == len(tasks):
                pd.DataFrame(rows).to_parquet(out)
                print(f"[imbalance] {i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(rows).to_parquet(out)


if __name__ == "__main__":
    main()
