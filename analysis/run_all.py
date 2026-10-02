#!/usr/bin/env python3
"""
run_all.py — regenerates every table, figure and statistic of the paper from the
run records.

Usage:
    python run_all.py                 # every step in order
    python run_all.py --from 4        # resume from step 4 (uses cached parquet)
    python run_all.py --only 8        # run only step 8

Set AL_RECORDS (run records) and optionally AL_OUT (outputs); see config.py.

Steps:
    1  load         read and check the run records          -> cache/long*.parquet
    2  saturation   early-budget window (n95) per setting    -> cache/saturation.parquet
    3  aulc         AULC per split on the window             -> cache/aulc_split.parquet
    4  leaderboard  ranks, regret, failure rates             -> cache/global_summary.parquet
    5  structure    indicators of the uncertain region       -> cache/delta_structure.parquet
    6  figures      Figures 1-5                              -> figures/
    7  tables       Tables 2, 4, 5, 6 and B.14 (LaTeX)        -> tables/
    8  difficulty   Table 7                                  -> tables/
    9  classifier   classifier shifts                        -> tables/
   10  trend        robustness of the null result (Sec. 4.3) -> console
   11  statistics   tests, windows, failure definitions, extra strategies, Fig. B.6
"""
import argparse
import sys
import time

import config as C

import step1_load
import step2_saturation
import step3_aulc
import step4_leaderboard
import step5_structure
import step6_figures
import step7_tables
import step8_difficulty
import step9_classifier
import step10_trend_robustness
import step11_statistics

STEPS = [
    (1, "load", step1_load.run),
    (2, "saturation", step2_saturation.run),
    (3, "aulc", step3_aulc.run),
    (4, "leaderboard", step4_leaderboard.run),
    (5, "structure", step5_structure.run),
    (6, "figures", step6_figures.run),
    (7, "tables", step7_tables.run),
    (8, "difficulty", step8_difficulty.run),
    (9, "classifier", step9_classifier.run),
    (10, "trend", step10_trend_robustness.run),
    (11, "statistics", step11_statistics.run),
]


def main():
    ap = argparse.ArgumentParser(description="Regenerate the analysis of the paper.")
    ap.add_argument("--from", dest="start", type=int, default=1, help="resume from this step")
    ap.add_argument("--only", type=int, default=None, help="run only this step")
    args = ap.parse_args()

    if not C.RECORDS.exists():
        sys.exit(f"[run_all] run records not found: {C.RECORDS} (set AL_RECORDS)")
    print(f"[run_all] records : {C.RECORDS}")
    print(f"[run_all] protocol: {C.PROTOCOL}")
    print(f"[run_all] outputs : {C.OUT_ROOT}\n")

    for num, name, fn in STEPS:
        if args.only is not None and num != args.only:
            continue
        if args.only is None and num < args.start:
            continue
        print(f"{'='*72}\n[run_all] STEP {num}: {name}\n{'='*72}")
        t0 = time.time()
        try:
            fn()
        except Exception as e:
            print(f"[run_all] STEP {num} ({name}) FAILED: {type(e).__name__}: {e}")
            sys.exit(1)
        print(f"[run_all] step {num} done in {time.time()-t0:.1f}s\n")

    print(f"[run_all] complete. Outputs in {C.OUT_ROOT}")


if __name__ == "__main__":
    main()
