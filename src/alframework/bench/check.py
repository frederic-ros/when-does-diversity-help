"""Integrity tools.

check_campaign : every expected run exists, is status=ok, has the current
                 protocol hash, the full budget grid, and all runs of a
                 (regime, dataset, split) share the same pairing key
                 (same subsample, train/test split and initial set) and the
                 same code commit. Returns a report; the analysis refuses to
                 start unless it is clean.
repro_test     : re-executes a random sample of runs and requires identical
                 learning curves and selections (result_hash).
export_long    : flattens all records into the long table used by the
                 analysis pipeline (one row per run x budget point).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .campaign import enumerate_jobs, record_path
from .data import dataset_n_init


def _load(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check_campaign(proto, phash, out, regimes, groups=("main", "extras"), **filters) -> dict:
    jobs = enumerate_jobs(proto, regimes, groups=groups, **filters)
    rep = {"expected": len(jobs), "missing": [], "failed": [], "stale_protocol": [],
           "bad_grid": [], "pairing_mismatch": [], "commits": set(), "dirty_code": 0,
           "commit_by_strategy": {}, "mixed_strategies": []}
    pairing = defaultdict(set)
    strat_commits = defaultdict(set)
    for j in jobs:
        rg, ds, c, s, st = j[:5]
        tag = f"{rg}/{ds}/{c}/split{s:02d}/{st}"
        p = record_path(out, rg, ds, c, s, st)
        if not p.exists():
            rep["missing"].append(tag); continue
        r = _load(p)
        if r.get("status") != "ok":
            rep["failed"].append(f"{tag}: {r.get('error', {}).get('message')}"); continue
        if r.get("protocol_hash") != phash:
            rep["stale_protocol"].append(tag); continue
        rc = proto["regimes"][rg]
        grid = [h["n_labeled"] for h in r["history"]]
        if grid != list(range(dataset_n_init(rc, ds), rc["max_budget"] + 1, rc["batch_size"])):
            rep["bad_grid"].append(tag)
        pairing[(rg, ds, s)].add(r["data"]["pairing_key"])
        rep["commits"].add(r["code"]["commit"])
        strat_commits[st].add(r["code"]["commit"])
        rep["dirty_code"] += bool(r["code"].get("dirty"))
    rep["pairing_mismatch"] = [f"{k[0]}/{k[1]}/split{k[2]:02d}" for k, v in pairing.items() if len(v) > 1]
    # A campaign may span several code versions only if every strategy's runs come
    # from a single version (e.g. one strategy was fixed and re-run in full).
    rep["commit_by_strategy"] = {k: sorted(v) for k, v in sorted(strat_commits.items())}
    rep["mixed_strategies"] = [k for k, v in rep["commit_by_strategy"].items() if len(v) > 1]
    rep["commits"] = sorted(c for c in rep["commits"] if c)
    rep["clean"] = (not rep["missing"] and not rep["failed"] and not rep["stale_protocol"]
                    and not rep["bad_grid"] and not rep["pairing_mismatch"]
                    and not rep["mixed_strategies"] and rep["dirty_code"] == 0)
    return rep


def format_report(rep: dict, max_items: int = 15) -> str:
    lines = [f"expected runs      : {rep['expected']}"]
    for k in ("missing", "failed", "stale_protocol", "bad_grid", "pairing_mismatch"):
        v = rep[k]
        lines.append(f"{k:19s}: {len(v)}")
        lines += [f"    {x}" for x in v[:max_items]]
        if len(v) > max_items:
            lines.append(f"    ... (+{len(v) - max_items})")
    lines.append(f"code commits       : {rep['commits'] or '-'}")
    if len(rep["commits"]) > 1:
        lines.append("  per strategy (each must have exactly one):")
        for k, v in rep.get("commit_by_strategy", {}).items():
            lines.append(f"    {k:24s} {', '.join(c[:7] for c in v)}")
    lines.append(f"strategies mixing versions: {len(rep['mixed_strategies'])}"
                 + (f" -> {rep['mixed_strategies']}" if rep["mixed_strategies"] else ""))
    lines.append(f"runs from dirty code: {rep['dirty_code']}")
    lines.append("STATUS: CLEAN" if rep["clean"] else "STATUS: NOT CLEAN — analysis must not start")
    return "\n".join(lines)


def repro_test(proto, phash, out, data_root, n=10, seed=0, regimes=None) -> list:
    from .run import execute
    files = sorted(p for p in Path(out).rglob("*.json") if p.name != "check_report.json")
    recs = [r for r in (_load(f) for f in files)
            if r.get("status") == "ok" and (not regimes or r["key"]["regime"] in regimes)]
    random.Random(seed).shuffle(recs)
    results = []
    for r in recs[:n]:
        k = r["key"]
        new = execute(proto, phash, k["regime"], k["dataset"], k["classifier"], k["split"],
                      k["strategy"], r["strategy_impl"]["registry_name"],
                      r["strategy_impl"]["params"], data_root,
                      cache_dir=str(Path(out) / ".data_cache"))
        results.append({"key": k, "identical": new.get("result_hash") == r.get("result_hash")})
    return results


def export_long(out, dest):
    import pandas as pd
    rows = []
    for f in sorted(Path(out).rglob("*.json")):
        r = _load(f)
        if r.get("status") != "ok":
            continue
        k = r["key"]
        for h in r["history"]:
            row = {"regime": k["regime"], "clf": k["classifier"], "dataset": k["dataset"],
                   "split": k["split"], "strategy": k["strategy"],
                   "pairing_key": r["data"]["pairing_key"], "protocol_hash": r["protocol_hash"],
                   "n_classes": r["data"]["n_classes"], "n_features": r["data"]["n_features"]}
            row.update({kk: v for kk, v in h.items() if kk != "selected"})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_parquet(dest)
    return df
