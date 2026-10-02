"""
step8_difficulty.py — intrinsic-difficulty stratification (paper Table 4).

Defines each context's difficulty by the macro-F1 ceiling reached at the full
budget (proxy for 'achievable with all data'), tertiles it WITHIN each regime to
de-confound difficulty from regime, then reports mean strategy rank per
Easy/Medium/Hard tier and writes the LaTeX table fragment.

Inputs:  out/cache/aulc_split.parquet, out/cache/long.parquet
Outputs: out/cache/difficulty.parquet
         out/tables/tab4_difficulty.tex
Run standalone:  python step8_difficulty.py
"""
import numpy as np
import pandas as pd

import config as C

WIN = "aulc_adaptive"


def _ceilings(df, metric):
    rows = []
    for (regime, clf, ds), g in df.groupby(["regime", "clf", "dataset"]):
        nmax = g["n_labeled"].max()
        tail = g[g["n_labeled"] == nmax].groupby("strategy")[metric].mean()
        rows.append({"regime": regime, "clf": clf, "dataset": ds,
                     "ceiling": float(tail.max())})
    return pd.DataFrame(rows)


def _within_regime_tiers(C_df):
    parts = []
    for regime, grp in C_df.groupby("regime"):
        grp = grp.copy()
        grp["tier"] = pd.qcut(grp["ceiling"].rank(method="first"), 3,
                              labels=["Hard", "Medium", "Easy"])
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def run():
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    df = pd.read_parquet(C.CACHE / "long.parquet")
    metric = C.PRIMARY_METRIC

    cei = _within_regime_tiers(_ceilings(df, metric))
    cei.to_parquet(C.CACHE / "difficulty.parquet")

    cell = (S[S.metric == metric]
            .groupby(["regime", "clf", "dataset", "strategy"])[WIN].mean().reset_index())
    cell["rank"] = cell.groupby(["regime", "clf", "dataset"])[WIN].rank(ascending=False)
    cell = cell.merge(cei[["regime", "clf", "dataset", "tier"]],
                      on=["regime", "clf", "dataset"])

    piv = (cell.groupby(["strategy", "tier"], observed=True)["rank"].mean()
           .unstack("tier")[["Easy", "Medium", "Hard"]])
    piv["delta"] = piv["Hard"] - piv["Easy"]
    piv = piv.sort_values("Hard")

    print("[step8] mean rank by intrinsic-difficulty tier (within-regime tertiles):")
    print(piv.round(2).rename(index=C.PRETTY).to_string())

    def f(x):
        return f"{x:.1f}"

    def fd(x):
        return f"{x:+.1f}"

    rows = [f"{C.PRETTY[i]} & {f(r.Easy)} & {f(r.Medium)} & {f(r.Hard)} & {fd(r.delta)} \\\\"
            for i, r in piv.iterrows()]
    (C.TABDIR / "tab4_difficulty.tex").write_text("\n".join(rows))
    print(f"[step8] wrote tab4_difficulty.tex -> {C.TABDIR}")
    return piv


if __name__ == "__main__":
    run()
