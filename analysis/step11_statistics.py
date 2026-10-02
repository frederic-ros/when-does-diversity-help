"""step11_statistics.py — statistical tests and robustness analyses (from cached tables).

Writes out/tables/stats_*.csv and out/figures/fig_cd.png:
  stats_windows.csv      mean rank under adaptive 90/95/99 %, fixed 10-30 %, full budget
  stats_regret.csv       regret distribution (mean = AULC ordering; median, R90, max, rel>5 %)
  stats_stats.txt        Friedman, Nemenyi CD, Wilcoxon-Holm among leaders + margin, effect sizes
  stats_collapse.csv     collapse-rate sensitivity (worst-k, relative-regret thresholds, < random)
  stats_classifier.csv   RF-LR rank shift per regime, both metrics
  stats_trend.txt        band-width trend: per-method gains, difficulty control,
                       mixed model with source-dataset random intercept, cluster bootstrap
  stats_router.txt       router vs its fixed modes (DBAL, R22, WALC) and their per-cell oracle
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, studentized_range, wilcoxon

import config as C

warnings.filterwarnings("ignore")
METRIC = "f1_macro"
_p = lambda s: C.PRETTY.get(s, s)


def _trap(x, y):
    if len(x) < 2:
        return np.nan
    span = x.max() - x.min()
    return float(np.trapezoid(y, x) / span) if span > 0 else np.nan


# --------------------------------------------------------------------------- #
def windows():
    L = pd.read_parquet(C.CACHE / "long.parquet")
    out = []
    for (rg, clf, ds), g in L.groupby(["regime", "clf", "dataset"]):
        ref = g[g.strategy == "margin"].groupby("n_labeled")[METRIC].mean()
        bmax, bmin = g.n_labeled.max(), g.n_labeled.min()
        nw = {}
        for f in (0.90, 0.95, 0.99):
            n = ref[ref >= f * ref.max()].index.min()
            nw[f] = n if n > bmin else bmax          # same fallback as step 3
        for (st, sp), gs in g.groupby(["strategy", "split"]):
            gs = gs.sort_values("n_labeled")
            x, y = gs.n_labeled.values.astype(float), gs[METRIC].values
            rec = dict(regime=rg, clf=clf, dataset=ds, strategy=st, split=sp)
            for f in nw:
                m = x <= nw[f]
                rec[f"ad{int(f * 100)}"] = _trap(x[m], y[m])
            m = (x >= 0.1 * bmax) & (x <= 0.3 * bmax)
            rec["fixed"] = _trap(x[m], y[m])
            rec["full"] = _trap(x, y)
            out.append(rec)
    W = pd.DataFrame(out)
    W.to_parquet(C.CACHE / "stats_windows.parquet")
    cols = ["ad90", "ad95", "ad99", "fixed", "full"]
    cell = W.groupby(["regime", "clf", "dataset", "strategy"])[cols].mean().reset_index()
    R = pd.DataFrame({c: cell.assign(r=cell.groupby(["regime", "clf", "dataset"])[c]
                                     .rank(ascending=False)).groupby("strategy").r.mean()
                      for c in cols}).sort_values("ad95")
    R.index = [_p(s) for s in R.index]
    R.round(2).to_csv(C.TABDIR / "stats_windows.csv")
    return W


def _cells(W, col="ad95"):
    return W.groupby(["regime", "clf", "dataset", "strategy"])[col].mean().unstack("strategy")


def regret_stats_collapse(W, lines):
    cell = _cells(W)
    best = cell.max(axis=1)
    reg = (-cell).add(best, axis=0)
    rel = reg.div(best, axis=0)
    same = list(reg.mean().sort_values().index) == list(cell.mean().sort_values(ascending=False).index)
    lines.append(f"mean-regret ordering identical to mean-AULC ordering: {same}")
    q = pd.DataFrame({"mean": reg.mean(), "median": reg.median(), "R90": reg.quantile(.9),
                      "max": reg.max(), "rel>5%": (rel > .05).mean()})
    q.index = [_p(s) for s in q.index]
    q.sort_values("mean").round(4).to_csv(C.TABDIR / "stats_regret.csv")

    rk = cell.rank(axis=1, ascending=True)
    col = pd.DataFrame({f"worst{k}": (rk <= k).mean() for k in (1, 2, 3, 5)})
    for t in (.02, .05, .10):
        col[f"rel>{int(t * 100)}%"] = (rel > t).mean()
    if "random" in cell:
        col["<random"] = cell.lt(cell["random"], axis=0).mean()
    col.index = [_p(s) for s in col.index]
    col.sort_values("worst3").round(3).to_csv(C.TABDIR / "stats_collapse.csv")

    # Friedman / Nemenyi / Wilcoxon-Holm, on cells and on datasets
    from statsmodels.stats.multitest import multipletests
    for label, M in (("cells", cell), ("datasets (clf-averaged)",
                                       cell.groupby(level=["regime", "dataset"]).mean())):
        M = M.dropna(axis=1)
        N, k = M.shape
        chi, p = friedmanchisquare(*[M[c] for c in M.columns])
        cd = studentized_range.ppf(.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
        r = M.rank(axis=1, ascending=False).mean().sort_values()
        lines.append(f"\n[{label}] N={N} k={k} Friedman chi2={chi:.1f} p={p:.2e} Nemenyi CD={cd:.2f}")
        lines.append("  within CD of best: " + ", ".join(_p(s) for s in r.index if r[s] - r.iloc[0] <= cd))
        cands = [s for s in ("dbal", "rank2022", "V58", "walc_fixed", "diversity_optimized_batch", "margin") if s in M]
        pairs = [(a, b) for i, a in enumerate(cands) for b in cands[i + 1:]]
        ps = [wilcoxon(M[a] - M[b]).pvalue for a, b in pairs]
        pa = multipletests(ps, method="holm")[1]
        for (a, b), pv in zip(pairs, pa):
            d = (M[a] - M[b]).median()
            lines.append(f"  {_p(a):16s} vs {_p(b):16s} median diff={d:+.4f} p_holm={pv:.2e}")
        if label == "cells":
            _cd_plot(r, cd)


def _cd_plot(r, cd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(r)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(-3.5, k + 4.5); ax.set_ylim(0, 1); ax.invert_xaxis(); ax.axis("off")
    ax.hlines(0.85, 1, k, color="k")
    for t in range(1, k + 1):
        ax.vlines(t, .85, .88, color="k"); ax.text(t, .91, str(t), ha="center", fontsize=8)
    half = (k + 1) // 2
    for i, (n, v) in enumerate(r.items()):
        left = i < half
        y = .75 - .075 * (i if left else i - half)
        ax.plot([v, v], [.85, y], "k", lw=.8)
        ax.plot([v, 1.0 if left else k], [y, y], "k", lw=.8)
        ax.text(.95 if left else k + .05, y, f"{_p(n)} ({v:.2f})",
                ha="left" if left else "right", va="center", fontsize=8)
    vals, cl = r.values, []
    for i in range(k):
        j = max(jj for jj in range(k) if vals[jj] - vals[i] <= cd)
        if j > i and not any(a <= i and b >= j for a, b in cl):
            cl.append((i, j))
    for n_, (i, j) in enumerate(cl):
        y = .80 - .012 * n_
        ax.plot([vals[i] - .05, vals[j] + .05], [y, y], lw=3, color="tab:red", alpha=.7)
    ax.plot([k, k - cd], [.98, .98], "k", lw=1.5)
    ax.text(k - cd / 2, 1.0, f"CD={cd:.2f}", ha="center", fontsize=8)
    plt.tight_layout(); plt.savefig(C.FIGDIR / "fig_cd.png", dpi=200, bbox_inches="tight"); plt.close()


def classifier_table():
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    out = {}
    for met in ("f1_macro", "balanced_accuracy"):
        s = S[S.metric == met]
        cell = s.groupby(["regime", "clf", "dataset", "strategy"]).aulc_adaptive.mean().reset_index()
        cell["rank"] = cell.groupby(["regime", "clf", "dataset"]).aulc_adaptive.rank(ascending=False)
        t = cell.groupby(["regime", "clf", "strategy"])["rank"].mean().unstack(["regime", "clf"])
        d = pd.DataFrame({rg: t[(rg, "rf")] - t[(rg, "lr")] for rg in s.regime.unique()})
        g = cell.groupby(["clf", "strategy"])["rank"].mean().unstack("clf")
        d["overall"] = g.rf - g.lr
        d.index = [_p(i) for i in d.index]
        out[met] = d
    pd.concat(out, axis=1).round(2).to_csv(C.TABDIR / "stats_classifier.csv")


def trend(W, lines):
    import statsmodels.formula.api as smf
    D = pd.read_parquet(C.CACHE / "delta_structure.parquet")
    cell = _cells(W).reset_index()
    fin = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    fin = fin[fin.metric == METRIC].groupby(["regime", "clf", "dataset", "strategy"]).final.mean()
    ceil = fin.groupby(["regime", "clf", "dataset"]).max().rename("ceiling").reset_index()
    D = D.merge(cell, on=["regime", "clf", "dataset"]).merge(ceil, on=["regime", "clf", "dataset"])
    D["source"] = np.where(D.regime == "latent",
                           D.dataset.str.extract(r"^(mnist|minst|fashion|fashin|cifar)", expand=False)
                           .replace({"minst": "mnist", "fashin": "fashion"}), D.dataset)
    zc = lambda c: D.groupby("regime")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
    band = "low_margin_frac"
    lines.append(f"n cells={len(D)}, independent sources={D.source.nunique()}")
    lines.append(f"spearman(band width, mean margin) = {spearmanr(D[band], D.mean_margin)[0]:+.3f}")
    lines.append(f"spearman(band width, ceiling)     = {spearmanr(D[band], D.ceiling)[0]:+.3f}")
    branch = [s for s in C.PUREDIV_BRANCH if s in D]
    D["purediv_mean"] = D[branch].mean(axis=1) - D["margin"]
    for m in branch + ["purediv_mean"]:
        g = D[m] - D["margin"] if m != "purediv_mean" else D[m]
        zg = g.groupby(D.regime).transform(lambda x: (x - x.mean()) / (x.std() + 1e-12))
        r, p = spearmanr(zc(band), zg)
        lines.append(f"  within-regime rho(band width, gain of {m}) = {r:+.2f} (p={p:.1e})")
    for ctrl in ("", " + ceiling"):
        m = smf.mixedlm(f"delta_purediv ~ {band} + regime{ctrl}", D, groups=D.source).fit()
        lines.append(f"  mixed model (source RE){ctrl:10s}: slope={m.params[band]:+.4f} "
                     f"SE={m.bse[band]:.4f} p={m.pvalues[band]:.2e}")
    rng = np.random.default_rng(0); srcs = D.source.unique(); bs = []
    zb, zg = zc(band), zc("delta_purediv")
    for _ in range(2000):
        pick = rng.choice(srcs, len(srcs))
        idx = np.concatenate([np.flatnonzero(D.source.values == s) for s in pick])
        bs.append(spearmanr(zb.values[idx], zg.values[idx])[0])
    lines.append("  cluster bootstrap 95%% CI (by source): [%.2f, %.2f]" % tuple(np.percentile(bs, [2.5, 97.5])))
    tcols = [c for c in D.columns if c.startswith("band_width_tau")]
    for c in tcols:
        r, p = spearmanr(zc(c), zg)
        lines.append(f"  tau sensitivity {c}: rho={r:+.2f} p={p:.1e}")


def router(W, lines):
    cell = _cells(W)
    if "V58" not in cell:
        ce = _cells_with_extras()
        if ce is not None:
            cell = ce
    modes = [s for s in ("dbal", "rank2022", "walc_fixed") if s in cell]
    if "V58" not in cell:
        lines.append("router not in panel"); return
    orc = cell[modes].max(axis=1)
    lines.append(f"modes compared: {modes}")
    lines.append(f"mean AULC router={cell.V58.mean():.4f} " +
                 " ".join(f"{m}={cell[m].mean():.4f}" for m in modes) + f" oracle={orc.mean():.4f}")
    for m in modes:
        lines.append(f"  router vs {m}: mean diff={(cell.V58 - cell[m]).mean():+.4f} "
                     f"Wilcoxon p={wilcoxon(cell.V58 - cell[m]).pvalue:.3f}")


def _cells_with_extras():
    path = C.CACHE / "long_with_extras.parquet"
    if not path.exists():
        return None
    L = pd.read_parquet(path)
    rows = []
    for (rg, clf, ds), g in L.groupby(["regime", "clf", "dataset"]):
        ref = g[g.strategy == "margin"].groupby("n_labeled")[METRIC].mean()
        n = ref[ref >= .95 * ref.max()].index.min()
        n = n if n > g.n_labeled.min() else g.n_labeled.max()
        for (st, sp), gs in g.groupby(["strategy", "split"]):
            gs = gs.sort_values("n_labeled"); x = gs.n_labeled.values.astype(float)
            m = x <= n
            rows.append(dict(regime=rg, clf=clf, dataset=ds, strategy=st,
                             aulc=_trap(x[m], gs[METRIC].values[m])))
    return pd.DataFrame(rows).groupby(["regime", "clf", "dataset", "strategy"]).aulc.mean().unstack()


def extras():
    """Each extra strategy inserted alone into the main panel (same window)."""
    from statsmodels.stats.multitest import multipletests
    cell = _cells_with_extras()
    if cell is None:
        return ["no extra strategies in the records"]
    lines = []
    main = [c for c in cell.columns if c not in C.EXTRAS]
    for ex in [e for e in C.EXTRAS if e in cell.columns]:
        M = cell[main + [ex]].dropna()
        rk = M.rank(axis=1, ascending=False)
        reg = (-M).add(M.max(axis=1), axis=0)
        fail = (M.rank(axis=1, ascending=True) <= 3).mean()
        lines.append(f"\n== {_p(ex)} inserted into the main panel ({M.shape[1]} strategies, {len(M)} settings)")
        by = rk[ex].groupby(level="regime").mean().round(2).to_dict()
        lines.append(f"  mean rank {rk[ex].mean():.2f} (main leaders: DBAL {rk['dbal'].mean():.2f}, "
                     f"Rank-2022 {rk['rank2022'].mean():.2f}, margin {rk['margin'].mean():.2f}); per regime {by}")
        lines.append(f"  regret mean {reg[ex].mean():.4f} R90 {reg[ex].quantile(.9):.4f} max {reg[ex].max():.4f}; fail {fail[ex]:.0%}")
        comps = [c for c in ("dbal", "rank2022", "margin") if c in M]
        ps = [wilcoxon(M[ex] - M[c]).pvalue for c in comps]
        pa = multipletests(ps, method="holm")[1]
        for c, pv in zip(comps, pa):
            lines.append(f"  vs {_p(c):10s}: median diff {(M[ex]-M[c]).median():+.4f}, p_holm {pv:.2e}")
    if {"dbal", "dbal_dualselect", "margin"} <= set(cell.columns):
        lines.append("\n== DBAL implementations (reconciliation with the DualSelect paper)")
        for rg, g in cell.groupby(level="regime"):
            d1 = (g["dbal"] - g["margin"]); d2 = (g["dbal_dualselect"] - g["margin"])
            lines.append(f"  {rg:9s} DBAL(this paper)-margin {d1.mean():+.4f} (p={wilcoxon(d1).pvalue:.3f}); "
                         f"DBAL(DualSelect impl.)-margin {d2.mean():+.4f} (p={wilcoxon(d2).pvalue:.3f})")
    return lines


def run():
    W = windows()
    lines = []
    regret_stats_collapse(W, lines)
    (C.TABDIR / "stats_stats.txt").write_text("\n".join(lines)); print("\n".join(lines))
    classifier_table()
    tl = []; trend(W, tl); (C.TABDIR / "stats_trend.txt").write_text("\n".join(tl)); print("\n".join(tl))
    rl = []; router(W, rl); (C.TABDIR / "stats_router.txt").write_text("\n".join(rl)); print("\n".join(rl))
    el = extras(); (C.TABDIR / "stats_extras.txt").write_text("\n".join(el)); print("\n".join(el))


if __name__ == "__main__":
    run()
