"""
step4_leaderboard.py — leaderboards, regret/collapse, paired flatness tests,
metric divergence, and the router-value analysis.

Inputs:  out/cache/aulc_split.parquet
Outputs: out/cache/global_summary.parquet
         (prints per-regime leaderboards, Wilcoxon flatness, router-vs-oracle)
Run standalone:  python step4_leaderboard.py
"""
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

import config as C

WIN = "aulc_adaptive"


def _cell_table(S, window, metric):
    c = (S[S.metric == metric]
         .groupby(["regime", "clf", "dataset", "strategy", "family"])[window]
         .mean().reset_index())
    grp = c.groupby(["regime", "clf", "dataset"])
    c["rank"] = grp[window].rank(ascending=False)
    c["best"] = grp[window].transform("max")
    c["regret"] = c["best"] - c[window]
    c["worst3"] = grp[window].rank(ascending=True) <= 3
    return c


def _global_summary(S):
    def tbl(metric):
        c = _cell_table(S, WIN, metric)
        return c.groupby(["strategy", "family"]).agg(
            mrank=("rank", "mean"),
            win=("rank", lambda r: (r == 1).mean()),
            mreg=("regret", "mean"),
            p90reg=("regret", lambda x: np.percentile(x, 90)),
            coll=("worst3", "mean")).reset_index()

    A = tbl("f1_macro").rename(columns={
        "mrank": "rank_f1", "mreg": "reg_f1", "coll": "coll_f1",
        "win": "win_f1", "p90reg": "p90_f1"})
    B = tbl("balanced_accuracy")[["strategy", "mrank", "win", "mreg", "coll"]].rename(
        columns={"mrank": "rank_ba", "win": "win_ba", "mreg": "reg_ba", "coll": "coll_ba"})
    G = A.merge(B, on="strategy").sort_values("rank_f1")
    G.to_parquet(C.CACHE / "global_summary.parquet")
    return G


def _flatness(S):
    """Paired Wilcoxon among top strategies vs each other and vs margin."""
    sub = S[S.metric == "f1_macro"]
    top = [t for t in ["dbal", "rank2022", "V58", "diversity_optimized_batch", "margin"]
           if t in set(sub.strategy)]
    out = []
    for a, b in itertools.combinations(top, 2):
        sig = tot = 0
        deltas = []
        for _, g in sub.groupby(["regime", "clf", "dataset"]):
            pa = g[g.strategy == a].set_index("split")[WIN]
            pb = g[g.strategy == b].set_index("split")[WIN]
            idx = pa.index.intersection(pb.index)
            if len(idx) < 6:
                continue
            da, db = pa.loc[idx].values, pb.loc[idx].values
            deltas.append(np.mean(da - db))
            tot += 1
            if np.allclose(da - db, 0):
                continue
            try:
                if wilcoxon(da, db).pvalue < 0.05:
                    sig += 1
            except ValueError:
                pass
        out.append((a, b, sig, tot, float(np.median(deltas)) if deltas else np.nan))
    return out


def _router_value(S):
    """V58 vs the fixed modes it routes between (dbal, rank2022) and their oracle."""
    cell = (S[S.metric == "f1_macro"]
            .groupby(["regime", "clf", "dataset", "strategy"])[WIN].mean()
            .unstack("strategy"))
    cell["oracle"] = cell[["dbal", "rank2022"]].max(axis=1)
    res = {}
    for col in ["V58", "dbal", "rank2022", "oracle", "margin"]:
        reg = cell["oracle"] - cell[col]
        res[col] = (cell[col].mean(), reg.mean(), np.percentile(reg, 90))
    within = (cell["oracle"] - cell["V58"] <= 0.002).mean()
    beats_both = ((cell["V58"] > cell["dbal"]) & (cell["V58"] > cell["rank2022"])).mean()
    return res, within, beats_both


def run():
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    G = _global_summary(S)
    print("[step4] GLOBAL leaderboard (adaptive window):")
    show = G[["strategy", "family", "rank_f1", "rank_ba", "win_f1", "reg_f1", "p90_f1", "coll_f1"]]
    print(show.round(3).to_string(index=False))

    print("\n[step4] Per-regime mean rank (f1, adaptive):")
    c = _cell_table(S, WIN, "f1_macro")
    piv = c.groupby(["strategy", "regime"])["rank"].mean().unstack("regime")
    piv["overall"] = piv.mean(axis=1)
    print(piv.sort_values("overall").round(2).to_string())

    print("\n[step4] Flatness — paired Wilcoxon among top strategies (f1):")
    for a, b, sig, tot, med in _flatness(S):
        print(f"   {a:>26} vs {b:<26} sig {sig:2d}/{tot:2d} "
              f"({sig/max(tot,1):.0%})  median Δ={med:+.4f}")

    if "V58" not in set(S.strategy):
        print("\n[step4] Router not in the main panel: see step 11, stats_router.txt")
        return G
    print("\n[step4] Router value (V58 vs fixed modes & oracle):")
    res, within, beats = _router_value(S)
    for k, (a, r, p90) in res.items():
        print(f"   {k:>9}: AULC {a:.4f} | regret-vs-oracle {r:+.4f} | p90 {p90:+.4f}")
    print(f"   V58 within 0.002 of best-of-(dbal,rank2022): {within:.0%} of cells")
    print(f"   V58 strictly beats both fixed modes:         {beats:.0%} of cells")
    return G


if __name__ == "__main__":
    run()
