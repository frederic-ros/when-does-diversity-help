"""
step3_aulc.py — split-level paired windowed AULC (2 windows x 2 metrics).

For each (context, strategy, split, metric) it integrates the learning curve over:
  - aulc_adaptive : up to the per-cell n95 (from step2)   [PRIMARY]
  - aulc_fixed    : the fixed 10-30% budget band            [robustness]
  - aulc_full     : the whole curve                         [reference]
Splits are paired across strategies within a cell (the benchmark fixes the seed
schedule identically per strategy), which licenses the paired tests downstream.

Inputs:  out/cache/long.parquet, out/cache/saturation.parquet
Outputs: out/cache/aulc_split.parquet
Run standalone:  python step3_aulc.py
"""
import numpy as np
import pandas as pd

import config as C


def _aulc(x, y):
    if len(x) < 2:
        return np.nan
    order = np.argsort(x)
    x = np.asarray(x)[order]
    y = np.asarray(y)[order]
    span = x.max() - x.min()
    if span <= 0:
        return np.nan
    trap = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    return float(trap / span)


def run() -> pd.DataFrame:
    df = pd.read_parquet(C.CACHE / "long.parquet")
    sat = pd.read_parquet(C.CACHE / "saturation.parquet")

    adaptive = {}
    for r in sat.itertuples():
        nwin = r.n95 if (r.n95 and r.n95 > 0) else r.budget_max
        adaptive[(r.regime, r.clf, r.dataset)] = (nwin, r.budget_max)

    rows = []
    for (regime, clf, ds), g in df.groupby(["regime", "clf", "dataset"]):
        nwin_ad, bmax = adaptive.get((regime, clf, ds), (None, g["n_labeled"].max()))
        if not nwin_ad or nwin_ad <= g["n_labeled"].min():
            # margin saturated at the very first budget point (a one-point
            # window has no area): fall back to the full budget, as documented
            nwin_ad = bmax
        nlo, nhi = 0.10 * bmax, 0.30 * bmax
        for (strat, split), gs in g.groupby(["strategy", "split"]):
            gs = gs.sort_values("n_labeled")
            x = gs["n_labeled"].values
            for metric in C.METRICS:
                y = gs[metric].values
                ma = x <= nwin_ad
                mf = (x >= nlo) & (x <= nhi)
                rows.append({
                    "regime": regime, "clf": clf, "dataset": ds, "split": split,
                    "strategy": strat, "family": C.FAMILY.get(strat, "other"),
                    "metric": metric,
                    "aulc_adaptive": _aulc(x[ma], y[ma]),
                    "aulc_fixed": _aulc(x[mf], y[mf]) if mf.sum() >= 2 else np.nan,
                    "aulc_full": _aulc(x, y),
                    "final": float(y[-1]), "nwin": nwin_ad, "bmax": bmax,
                })
    S = pd.DataFrame(rows)
    out = C.CACHE / "aulc_split.parquet"
    S.to_parquet(out)
    print(f"[step3] {len(S):,} split-level rows, "
          f"{S.groupby(['regime','clf','dataset']).ngroups} cells -> {out}")
    return S


if __name__ == "__main__":
    run()
