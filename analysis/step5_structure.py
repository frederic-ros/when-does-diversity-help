"""
step5_structure.py — structural indicators of the uncertain region (paper Sec. 4.3).

The indicators are read from the run records: the benchmark logs them at round 0
(after training the initial model, before any query). They are averaged over the
splits of the margin runs (round 0 is identical for every strategy of a split,
since the initial set is shared). The gains over margin come from step 3.

Inputs:  out/cache/long_full.parquet (step 1), out/cache/aulc_split.parquet (step 3)
Outputs: out/cache/structure_indicators.parquet, out/cache/delta_structure.parquet
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C


def build_indicators():
    L = pd.read_parquet(C.CACHE / "long_full.parquet")
    r0 = L[(L["round"] == 0) & (L["strategy"] == "margin")]
    cols = [c for c in r0.columns if c.startswith(("band_width", "margin_m", "unc_"))] + ["n_classes", "n_features"]
    ST = r0.groupby(["regime", "dataset", "clf"])[cols].mean().reset_index()
    ST = ST.rename(columns={"band_width_tau0.1": "low_margin_frac", "margin_mean": "mean_margin",
                            "unc_eff_dim": "eff_dim_unc", "unc_clusterability": "clusterability"})
    ST.to_parquet(C.CACHE / "structure_indicators.parquet")
    return ST


def _build_deltas(ST):
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    out = []
    for regime in ["synthetic", "tabular", "latent"]:
        sub = S[(S.regime == regime) & (S.metric == "f1_macro")]
        cell = (sub.groupby(["dataset", "clf", "strategy"])["aulc_adaptive"]
                .mean().unstack("strategy"))
        clus = cell[C.CLUSTERING_BRANCH].max(axis=1)
        divp = cell[C.PUREDIV_BRANCH].max(axis=1)
        d = pd.DataFrame({"delta_clustering": clus - cell["margin"],
                          "delta_purediv": divp - cell["margin"]}).reset_index()
        d["regime"] = regime
        out.append(d)
    D = pd.concat(out, ignore_index=True).merge(
        ST, on=["regime", "dataset", "clf"], how="inner")
    D.to_parquet(C.CACHE / "delta_structure.parquet")
    return D


def _report(D):
    feats = ["low_margin_frac", "mean_margin", "eff_dim_unc", "clusterability", "n_classes"]

    def sp(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return ("--", 1.0)
        r, p = spearmanr(x, y)
        return (f"{r:+.2f}", p)

    print("\n[step5] Pure-diversity gain vs structure (Spearman rho), per regime + pooled:")
    print(f"   {'indicator':>16} | synth   tab     latent  pooled")
    for f in feats:
        cells = []
        for rg in ["synthetic", "tabular", "latent"]:
            s = D[D.regime == rg]
            cells.append(sp(s[f], s["delta_purediv"])[0])
        pooled = sp(D[f], D["delta_purediv"])[0]
        print(f"   {f:>16} | {cells[0]:>6} {cells[1]:>6} {cells[2]:>6}  {pooled:>6}")

    # within-regime standardized pooled (the robust law)
    def wz(df, col):
        return df.groupby("regime")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    Dz = D.copy()
    print("\n[step5] Within-regime standardized pooled (pure-diversity):")
    for f in feats:
        if D.groupby("regime")[f].std().min() == 0:
            print(f"   {f:>16}: -- (constant within a regime)")
            continue
        rho, p = spearmanr(wz(Dz, f), wz(Dz, "delta_purediv"))
        print(f"   {f:>16}: rho={rho:+.3f}  p={p:.3f}")


def run():
    ST = build_indicators()
    D = _build_deltas(ST)
    _report(D)
    return D


if __name__ == "__main__":
    run()
