"""
step9_classifier.py — classifier-effect analysis (paper Table 5).

Compares mean strategy rank under logistic regression vs random forest and the
rank shift RF-LR, for both metrics; writes the LaTeX table fragment for macro-F1.
Uncertainty methods improve under RF; pure-diversity/density degrade; integrated
clustering and the router are classifier-robust.

Inputs:  out/cache/aulc_split.parquet
Outputs: out/tables/tab5_classifier.tex
Run standalone:  python step9_classifier.py
"""
import numpy as np
import pandas as pd

import config as C

WIN = "aulc_adaptive"


def _rank_by_clf(S, metric):
    cell = (S[S.metric == metric]
            .groupby(["regime", "clf", "dataset", "strategy"])[WIN].mean().reset_index())
    cell["rank"] = cell.groupby(["regime", "clf", "dataset"])[WIN].rank(ascending=False)
    piv = cell.groupby(["strategy", "clf"])["rank"].mean().unstack("clf")
    piv["delta"] = piv["rf"] - piv["lr"]
    return piv.sort_values("delta")


def run():
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")

    for metric in C.METRICS:
        piv = _rank_by_clf(S, metric)
        print(f"\n[step9] mean rank by classifier ({metric}), sorted by RF-LR shift:")
        print(piv.round(2).rename(index=C.PRETTY).to_string())

    # family-level shift (macro-F1) for the mechanism discussion
    piv = _rank_by_clf(S, C.PRIMARY_METRIC).reset_index()
    piv["family"] = piv["strategy"].map(C.FAMILY)
    print("\n[step9] mean RF-LR shift by family (macro-F1):")
    print(piv.groupby("family")["delta"].mean().round(2).sort_values().to_string())

    # LaTeX fragment (macro-F1)
    piv = _rank_by_clf(S, C.PRIMARY_METRIC)

    def f(x):
        return f"{x:.1f}"

    def fd(x):
        return f"{x:+.1f}"

    rows = [f"{C.PRETTY[i]} & {f(r.lr)} & {f(r.rf)} & {fd(r.delta)} \\\\"
            for i, r in piv.iterrows()]
    (C.TABDIR / "tab5_classifier.tex").write_text("\n".join(rows))
    print(f"\n[step9] wrote tab5_classifier.tex -> {C.TABDIR}")
    return piv


if __name__ == "__main__":
    run()
