"""
step10_trend_robustness.py — defensive statistics for the cross-regime trend
(paper Section 4.3, the 'is this a law or a pooled artifact?' paragraph).

Three converging checks that the mean-margin / low-margin-fraction effect on
pure-diversity gain survives regime control:
  (1) partial Spearman (within-regime z-scoring of both variables),
  (2) linear mixed-effects model  delta ~ indicator + (1|regime),
  (3) leave-one-regime-out pooled Spearman.

Inputs:  out/cache/delta_structure.parquet   (from step5)
Outputs: console report (numbers quoted in the paper)
Run standalone:  python step10_trend_robustness.py
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C

warnings.filterwarnings("ignore")

KEY = ["low_margin_frac", "mean_margin"]   # the indicators claimed to generalize


def _within_z(df, col):
    return df.groupby("regime")[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))


def run():
    fp = C.CACHE / "delta_structure.parquet"
    if not fp.exists():
        print("[step10] delta_structure.parquet missing — run step5 first.")
        return
    M = pd.read_parquet(fp)

    print("[step10] (1) Partial Spearman, regime removed (within-regime z-scored):")
    for f in KEY + ["eff_dim_unc", "clusterability"]:
        rho, p = spearmanr(_within_z(M, f), _within_z(M, "delta_purediv"))
        print(f"     {f:>16}: rho={rho:+.3f}  p={p:.4f}  n={len(M)}")

    print("\n[step10] (2) Linear mixed-effects: delta ~ indicator + (1|regime):")
    try:
        import statsmodels.formula.api as smf
        for f in KEY:
            d = M[["delta_purediv", "regime", f]].dropna().rename(columns={f: "x"})
            d["x"] = (d["x"] - d["x"].mean()) / d["x"].std()
            res = smf.mixedlm("delta_purediv ~ x", d, groups=d["regime"]).fit(
                reml=True, method="lbfgs")
            print(f"     {f:>16}: slope={res.params['x']:+.4f} "
                  f"(SE {res.bse['x']:.4f})  p={res.pvalues['x']:.4f}")
    except Exception as e:
        print(f"     statsmodels unavailable or failed ({e}); "
              f"the partial Spearman and leave-one-out carry the argument.")

    print("\n[step10] (3) Leave-one-regime-out pooled Spearman:")
    for f in KEY:
        print(f"     {f}:")
        for drop in ["synthetic", "tabular", "latent"]:
            sub = M[M.regime != drop]
            rho, p = spearmanr(sub[f], sub["delta_purediv"])
            print(f"        without {drop:>9}: rho={rho:+.3f}  p={p:.4f}  (n={len(sub)})")


if __name__ == "__main__":
    run()
