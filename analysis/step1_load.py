"""
step1_load.py — read every run record into one long table.

Checks the campaign first (completeness, pairing hashes, code versions) and refuses
to continue if it is not clean.

Outputs: out/cache/long_full.parquet (all strategies, all logged fields),
         out/cache/long_with_extras.parquet, out/cache/long.parquet (17-strategy panel)
"""
import pandas as pd

import config as C


def run() -> pd.DataFrame:
    """Read the run records. Refuses to load an incomplete or inconsistent campaign."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from alframework.bench.campaign import load_protocol
    from alframework.bench.check import check_campaign, export_long, format_report
    proto, phash = load_protocol(str(C.PROTOCOL))
    rep = check_campaign(proto, phash, C.RECORDS, list(proto["regimes"]))
    if not rep["clean"]:
        print(format_report(rep))
        raise SystemExit("[step1] campaign is not clean; fix it before analysing.")
    df = export_long(C.RECORDS, C.CACHE / "long_full.parquet")
    df["strategy"] = df["strategy"].map(C.norm_strat)
    df["family"] = df["strategy"].map(lambda s: C.FAMILY.get(s, "other"))
    keep = ["regime", "clf", "dataset", "split", "strategy", "family", "n_labeled",
            "accuracy", "f1_macro", "balanced_accuracy", "f1_weighted"]
    out = df[keep]
    out.to_parquet(C.CACHE / "long_with_extras.parquet")
    out = out[~out.strategy.isin(C.EXTRAS)]
    out.to_parquet(C.CACHE / "long.parquet")
    print(f"[step1] {len(out):,} rows from a clean campaign (protocol {phash})")
    return out



if __name__ == "__main__":
    run()
