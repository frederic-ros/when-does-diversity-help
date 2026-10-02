"""Campaign orchestration: enumerate runs, skip completed ones, run in parallel.

Output layout (one JSON per run, written atomically):
    <out>/<regime>/<dataset>/<classifier>/split<NN>/<strategy>.json
A run is considered done iff its file exists, status == "ok" and its
protocol_hash equals the current one. Anything else is (re)run.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

from .provenance import hash_obj


def load_protocol(path: str) -> tuple[dict, str]:
    with open(path, encoding="utf-8") as f:
        proto = json.load(f)
    # the hash covers everything that affects results (not free-text notes)
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k not in ("note", "description", "source")}
        return o
    return proto, hash_obj(strip(proto))


def panel(proto: dict, groups: Iterable[str]) -> dict:
    out = {}
    for g in groups:
        for name, (reg, params) in proto["panel"][g].items():
            out[name] = (reg, params)
    return out


def record_path(out: Path, regime, dataset, clf, split, strategy) -> Path:
    return Path(out) / regime / dataset / clf / f"split{split:02d}" / f"{strategy}.json"


def enumerate_jobs(proto: dict, regimes, datasets=None, classifiers=None, splits=None,
                   strategies=None, groups=("main", "extras")) -> List[tuple]:
    pan = panel(proto, groups)
    if strategies:
        unknown = set(strategies) - set(pan)
        if unknown:
            raise KeyError(f"unknown strategies: {sorted(unknown)}")
        pan = {k: v for k, v in pan.items() if k in strategies}
    clfs = classifiers or list(proto["classifiers"])
    jobs = []
    for rg in regimes:
        rc = proto["regimes"][rg]
        dss = [d for d in rc["datasets"] if not datasets or d in datasets]
        sps = splits if splits is not None else range(rc["splits"])
        for ds in dss:
            for c in clfs:
                for s in sps:
                    if s >= rc["splits"]:
                        raise ValueError(f"split {s} outside protocol ({rc['splits']} splits)")
                    for st, (reg, params) in pan.items():
                        jobs.append((rg, ds, c, int(s), st, reg, params))
    return jobs


def is_done(path: Path, protocol_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
        return r.get("status") == "ok" and r.get("protocol_hash") == protocol_hash
    except Exception:
        return False


def write_atomic(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, separators=(",", ":"))
    os.replace(tmp, path)


def _worker(args):
    # single-threaded numerics for bit-exact reproducibility
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    import warnings
    warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")
    from .run import execute
    proto, phash, out, data_root, job = args
    rg, ds, c, s, st, reg, params = job
    rec = execute(proto, phash, rg, ds, c, s, st, reg, params, data_root,
                  cache_dir=str(Path(out) / ".data_cache"))
    write_atomic(record_path(out, rg, ds, c, s, st), rec)
    return job, rec["status"], rec.get("wall_time_s", 0.0), rec.get("error", {}).get("message")


def run_campaign(proto, phash, jobs, out, data_root: Optional[str], n_jobs: int = 1,
                 force: bool = False, log=print) -> dict:
    out = Path(out)
    todo = [j for j in jobs if force or not is_done(record_path(out, *j[:5]), phash)]
    log(f"[campaign] {len(jobs)} runs in scope, {len(jobs) - len(todo)} already done, "
        f"{len(todo)} to run with {n_jobs} worker(s)")
    # parse every real dataset once, sequentially, before forking workers
    from .data import load_txt
    for rg, ds in sorted({(j[0], j[1]) for j in todo if j[0] != "synthetic"}):
        rc = proto["regimes"][rg]
        load_txt(str(Path(data_root) / rc["data_dir"] / rc["datasets"][ds]),
                 str(out / ".data_cache"))
    stats = {"ok": 0, "failed": 0}
    args = [(proto, phash, str(out), data_root, j) for j in todo]
    if n_jobs <= 1:
        it = map(_worker, args)
    else:
        ex = ProcessPoolExecutor(max_workers=n_jobs)
        it = (f.result() for f in as_completed([ex.submit(_worker, a) for a in args]))
    for i, (job, status, wt, err) in enumerate(it, 1):
        stats[status] = stats.get(status, 0) + 1
        tag = "/".join(map(str, job[:5]))
        if status != "ok":
            log(f"  [FAILED] {tag}: {err}")
        elif i % 50 == 0 or i == len(todo):
            log(f"  [{i}/{len(todo)}] last: {tag} ({wt:.1f}s)")
    if n_jobs > 1:
        ex.shutdown()
    log(f"[campaign] done: {stats}")
    return stats
