#!/usr/bin/env python3
"""Benchmark command-line interface.

    # run (resumable; completed runs with the same protocol hash are skipped)
    python benchmarks/run_benchmark.py run --regime synthetic tabular latent \
        --data-root /path/to/zenodo --out records --jobs 8

    # integrity: completeness, failures, budget grid, pairing, single commit
    python benchmarks/run_benchmark.py check --regime synthetic tabular latent --out records

    # bit-exact re-execution of a random sample of stored runs
    python benchmarks/run_benchmark.py repro --out records --data-root ... -n 20

    # flatten to the long table consumed by the analysis pipeline
    python benchmarks/run_benchmark.py export --out records --dest long.parquet

Filters (--datasets, --classifiers, --splits, --strategies, --groups) allow
smoke tests and sharding a campaign across machines; the protocol itself is
only defined by configs/protocol.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alframework.bench.campaign import enumerate_jobs, load_protocol, run_campaign  # noqa: E402
from alframework.bench.check import (check_campaign, export_long, format_report,  # noqa: E402
                                     repro_test)

DEFAULT_PROTO = ROOT / "configs" / "protocol.json"


def _filters(a):
    return dict(datasets=a.datasets, classifiers=a.classifiers,
                splits=a.splits, strategies=a.strategies)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "check", "repro", "export", "plan"])
    ap.add_argument("--protocol", default=str(DEFAULT_PROTO))
    ap.add_argument("--out", default="records")
    ap.add_argument("--data-root", default=None, help="root of the Zenodo data (tab/, latent/)")
    ap.add_argument("--regime", nargs="+", default=["synthetic", "tabular", "latent"])
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--classifiers", nargs="+")
    ap.add_argument("--splits", nargs="+", type=int)
    ap.add_argument("--strategies", nargs="+")
    ap.add_argument("--groups", nargs="+", default=["main", "extras"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="re-run even completed runs")
    ap.add_argument("-n", type=int, default=10, help="repro: number of runs to re-execute")
    ap.add_argument("--dest", default="long.parquet")
    a = ap.parse_args()

    proto, phash = load_protocol(a.protocol)
    print(f"[benchmark] protocol {proto['protocol_version']} hash={phash}")

    if a.command == "plan":
        jobs = enumerate_jobs(proto, a.regime, groups=a.groups, **_filters(a))
        print(f"{len(jobs)} runs")
        return
    if a.command == "run":
        jobs = enumerate_jobs(proto, a.regime, groups=a.groups, **_filters(a))
        st = run_campaign(proto, phash, jobs, a.out, a.data_root, a.jobs, a.force)
        sys.exit(1 if st.get("failed") else 0)
    if a.command == "check":
        rep = check_campaign(proto, phash, a.out, a.regime, groups=a.groups, **_filters(a))
        print(format_report(rep))
        Path(a.out).mkdir(parents=True, exist_ok=True)
        with open(Path(a.out) / "check_report.json", "w") as f:
            json.dump(rep, f, indent=1, default=list)
        sys.exit(0 if rep["clean"] else 2)
    if a.command == "repro":
        res = repro_test(proto, phash, a.out, a.data_root, n=a.n, regimes=a.regime)
        bad = [r for r in res if not r["identical"]]
        for r in res:
            print(("  OK   " if r["identical"] else "  DIFF ") + "/".join(map(str, r["key"].values())))
        print(f"[repro] {len(res) - len(bad)}/{len(res)} identical")
        sys.exit(1 if bad else 0)
    if a.command == "export":
        df = export_long(a.out, a.dest)
        print(f"[export] {len(df):,} rows -> {a.dest}")


if __name__ == "__main__":
    main()
