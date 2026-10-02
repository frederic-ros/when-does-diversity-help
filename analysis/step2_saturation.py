"""
step2_saturation.py — characterize saturation and define ante-saturation windows.

For every context (regime, clf, dataset) it computes, using margin as the
reference learner, the budget point n95 where margin first reaches 95% of its own
plateau. This defines the per-cell adaptive window used downstream. It also dumps
a saturation summary.

Inputs:  out/cache/long.parquet
Outputs: out/cache/saturation.parquet
Run standalone:  python step2_saturation.py
"""
import numpy as np
import pandas as pd

import config as C


def run() -> pd.DataFrame:
    df = pd.read_parquet(C.CACHE / "long.parquet")
    metric = C.PRIMARY_METRIC
    recs = []
    for (regime, clf, ds), g in df.groupby(["regime", "clf", "dataset"]):
        grid = sorted(g["n_labeled"].unique())
        piv = g.groupby(["strategy", "n_labeled"])[metric].mean().unstack("n_labeled")
        ref = piv.loc["margin"] if "margin" in piv.index else piv.mean()

        def first_frac(curve, frac):
            thr = frac * curve.max()
            reached = curve[curve >= thr]
            return None if len(reached) == 0 else reached.index.min()

        bmax = max(grid)
        n95 = first_frac(ref, 0.95)
        recs.append({
            "regime": regime, "clf": clf, "dataset": ds,
            "n_points": len(grid), "budget_max": bmax,
            "ref_start": float(ref.iloc[0]), "ref_max": float(ref.max()),
            "gain_total": float(ref.max() - ref.iloc[0]),
            "n95": n95,
            "frac_budget_to_95": (n95 / bmax) if n95 else None,
        })
    sat = pd.DataFrame(recs)
    out = C.CACHE / "saturation.parquet"
    sat.to_parquet(out)
    print(f"[step2] {len(sat)} context cells -> {out}")
    print("[step2] mean gain & fraction-of-budget-to-95% by regime:")
    print(sat.groupby("regime")[["gain_total", "frac_budget_to_95", "budget_max"]]
          .mean().round(3).to_string())
    return sat


if __name__ == "__main__":
    run()
