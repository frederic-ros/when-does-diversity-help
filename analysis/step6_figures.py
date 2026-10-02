"""
step6_figures.py — generate the three paper figures.

  fig_spread.png    : across-strategy spread vs budget fraction (ante-saturation)
  fig_curves.png    : representative learning curves with the window shaded
  fig_structure.png : pure-diversity gain vs structural indicators (3 regimes)
  fig_risk.png      : risk-vs-performance summary (mean rank vs collapse rate)
Inputs:  out/cache/long.parquet, saturation.parquet, delta_structure.parquet
Outputs: out/figures/*.png
Run standalone:  python step6_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import config as C

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})
REGIME_COL = {"synthetic": "#2ca02c", "tabular": "#1f77b4", "latent": "#d62728"}

# Some dataset names have aliases; figures accept either and use whichever is
# present in the data.
DS_ALIASES = {"covertype_original": "covertype", "letter_recognition_original": "letter_recognition",
              "rice_original": "rice", "cifar10trainpca100": "cifar10_pca100",
              "minsttrainpca100": "mnist_pca100", "minsttrainpca10": "mnist_pca10"}


def _resolve(ds, present_ds):
    if ds in present_ds:
        return ds
    alt = DS_ALIASES.get(ds)
    return alt if alt in present_ds else ds


def fig_spread():
    """Paper Fig. 1: standard deviation of macro-F1 across the 17 strategies (mean
    curve over splits), averaged over settings, versus the fraction of the labeling
    budget spent after the initial set."""
    df = pd.read_parquet(C.CACHE / "long.parquet")
    sep = []
    for (regime, clf, ds), g in df.groupby(["regime", "clf", "dataset"]):
        m = g.groupby(["strategy", "n_labeled"])[C.PRIMARY_METRIC].mean().unstack("strategy")
        n0, bmax = m.index.min(), m.index.max()
        for n, row in m.iterrows():
            sep.append({"regime": regime, "bf": (n - n0) / (bmax - n0), "sd": row.std()})
    sep = pd.DataFrame(sep)
    lab = {"synthetic": "Synthetic", "tabular": "Tabular", "latent": "Embeddings"}
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for r, c in REGIME_COL.items():
        s = sep[sep.regime == r]
        b = pd.cut(s.bf, np.linspace(-1e-9, 1, 11))
        gp = s.groupby(b, observed=True)["sd"].mean()
        ax.plot([iv.mid for iv in gp.index], gp.values, "o-", color=c,
                label=lab.get(r, r), lw=2, ms=5)
    ax.set_xlabel("Fraction of the labeling budget spent", fontsize=11)
    ax.set_ylabel("Spread between strategies\n(s.d. of macro-F1 across the 17 strategies)",
                  fontsize=10)
    ax.grid(alpha=0.3); ax.legend(frameon=False); ax.set_ylim(0, None)
    fig.tight_layout(); fig.savefig(C.FIGDIR / "fig_spread.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[step6] fig_spread.png")


def fig_curves():
    df = pd.read_parquet(C.CACHE / "long.parquet")
    sat = pd.read_parquet(C.CACHE / "saturation.parquet")
    groups = {"margin": ("#1f77b4", "-"), "dbal": ("#d62728", "-"),
              "rank2022": ("#ff7f0e", "-"), "V58": ("#2ca02c", "-"),
              "random": ("#999999", "--"), "probcover": ("#9467bd", ":"),
              "robust_qbc": ("#8c564b", ":")}

    CURVE_NAME = {"margin": "Margin", "dbal": "DBAL", "rank2022": "Rank-2022",
                  "random": "Random", "probcover": "ProbCover", "robust_qbc": "Robust QBC"}
    REG = {"synthetic": "Synthetic", "tabular": "Tabular", "latent": "Embeddings"}

    def panel(ax, regime, clf, ds):
        g = df[(df.regime == regime) & (df.clf == clf) & (df.dataset == ds)]
        if g.empty:
            ax.set_visible(False); return
        for st, (col, ls) in groups.items():
            s = g[g.strategy == st].groupby("n_labeled")[C.PRIMARY_METRIC].mean()
            if s.empty:
                continue
            ax.plot(s.index, s.values, color=col, ls=ls, label=CURVE_NAME.get(st, st), lw=1.6)
        row = sat[(sat.regime == regime) & (sat.clf == clf) & (sat.dataset == ds)]
        if len(row) and row.n95.values[0]:
            ax.axvline(row.n95.values[0], color="k", ls="--", alpha=0.4, lw=1)
            ax.axvspan(g.n_labeled.min(), row.n95.values[0], color="gold", alpha=0.07)
        ax.set_title(f"{REG.get(regime, regime)} / {clf.upper()} / {ds}", fontsize=9)
        ax.set_xlabel("Number of labeled points"); ax.grid(alpha=0.3)

    # pick representative cells that exist in the data
    candidates = [("synthetic", "rf", "many_classes"), ("synthetic", "lr", "clean20"),
                  ("tabular", "rf", "letter_recognition_original"), ("tabular", "lr", "rice_original"),
                  ("latent", "lr", "cifar10trainpca100"), ("latent", "rf", "minsttrainpca10")]
    present = set(map(tuple, df[["regime", "clf", "dataset"]].drop_duplicates().values))
    present_ds = {d for _, _, d in present}
    candidates = [(r, c, _resolve(d, present_ds)) for r, c, d in candidates]
    cells = [c for c in candidates if c in present][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, cell in zip(axes.flat, cells):
        panel(ax, *cell)
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(); fig.savefig(C.FIGDIR / "fig_curves.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[step6] fig_curves.png")


def fig_structure():
    """Paper Fig. 5: gain of pure diversity over margin versus the three indicators of
    Table 6, one point per setting. Panel titles give the pooled Spearman correlation
    after within-regime standardisation (as in Table 6); dashed: pooled linear fit."""
    fp = C.CACHE / "delta_structure.parquet"
    if not fp.exists():
        print("[step6] skipping fig_structure (run step 5 first)")
        return
    M = pd.read_parquet(fp)
    lab = {"synthetic": "Synthetic", "tabular": "Tabular", "latent": "Embeddings"}
    panels = [("low_margin_frac", r"Uncertain fraction $w_{0.1}$"),
              ("eff_dim_unc", "Effective dimension of the uncertain points"),
              ("clusterability", "Clusterability of the uncertain points")]
    wz = lambda col: M.groupby("regime")[col].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (xf, xlabel) in zip(axes, panels):
        for r, c in REGIME_COL.items():
            sub = M[M.regime == r]
            ax.scatter(sub[xf], sub["delta_purediv"], c=c, label=lab[r], alpha=0.6, s=30,
                       edgecolor="white", linewidth=0.5)
        rho, p = spearmanr(wz(xf), wz("delta_purediv"))
        z = np.polyfit(M[xf], M["delta_purediv"], 1)
        xs = np.linspace(M[xf].min(), M[xf].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "k--", alpha=0.5, lw=1)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_title(f"within-regime Spearman $\\rho$ = {rho:+.2f} (p = {p:.2f})", fontsize=11)
        ax.tick_params(labelsize=10); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Gain of pure diversity over margin (AULC)", fontsize=11)
    axes[0].legend(fontsize=10, frameon=False)
    fig.tight_layout(); fig.savefig(C.FIGDIR / "fig_structure.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[step6] fig_structure.png")


def fig_risk():
    """Summary figure (paper Fig. 2): mean rank (x) vs failure rate (y), point size
    proportional to mean regret, colour = family. Labels are placed with explicit
    offsets and thin leader lines so that the dense lower-left region stays legible."""
    from matplotlib.lines import Line2D
    G = pd.read_parquet(C.CACHE / "global_summary.parquet")
    G = G[G.family != "router"]
    FAMC = {"uncertainty": "#1f77b4", "diversity": "#d62728", "hybrid": "#9467bd",
            "committee": "#8c564b", "clustering": "#2ca02c", "random": "#7f7f7f"}
    FAML = {"clustering": "Integrated clustering", "uncertainty": "Uncertainty",
            "hybrid": "Hybrid", "committee": "Committee",
            "diversity": "Pure diversity", "random": "Random"}
    # label offsets (dx in rank units, dy in failure-rate units) from each point
    OFF = {"dbal": (-0.55, 0.11), "rank2022": (0.05, 0.16),
           "unc_feature_kmeans": (-0.35, -0.085), "diversity_optimized_batch": (0.15, 0.10),
           "margin": (0.0, -0.085), "qbc": (0.0, -0.085), "badge_approx": (0.55, -0.065),
           "least_confident": (-0.95, 0.045), "tri_committee": (-1.0, 0.05),
           "coreset_kmeanspp": (0.75, -0.04), "entropy": (-0.55, 0.06),
           "adaptive_disagreement": (0.1, 0.085), "random": (0.95, -0.03),
           "coreset_greedy": (-1.2, 0.07), "typiclust": (0.9, -0.05),
           "probcover": (-1.3, 0.05), "robust_qbc": (0.9, -0.07)}
    NAME = dict(C.PRETTY, unc_feature_kmeans="Unc.-aug. $k$-means",
                adaptive_disagreement="Adapt. disagreement",
                coreset_kmeanspp="Core-set ($k$-means++)")
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for r in G.itertuples():
        ax.scatter(r.rank_f1, r.coll_f1, s=40 + r.reg_f1 * 9000,
                   color=FAMC.get(r.family, "#333"), alpha=0.85,
                   edgecolor="white", linewidth=1.2, zorder=3)
    for r in G.itertuples():
        dx, dy = OFF.get(r.strategy, (0.3, 0.04))
        ha = "center" if abs(dx) < 0.2 else ("left" if dx > 0 else "right")
        ax.annotate(NAME[r.strategy], xy=(r.rank_f1, r.coll_f1),
                    xytext=(r.rank_f1 + dx, r.coll_f1 + dy), fontsize=8.5,
                    ha=ha, va="center", color="#222", zorder=4,
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.6,
                                    shrinkA=1, shrinkB=4))
    n_strat = len(G)
    ax.set_xlabel("Mean rank over the 86 settings (lower is better)", fontsize=10.5)
    ax.set_ylabel(f"Failure rate (share of settings among the 3 worst of {n_strat})",
                  fontsize=10.5)
    ax.set_xlim(2.0, 17); ax.set_ylim(-0.13, 0.86); ax.grid(alpha=0.25, zorder=0)
    fam_handles = [Line2D([0], [0], marker="o", ls="", color=FAMC[k], label=FAML[k],
                          ms=9, mec="white") for k in FAML]
    leg1 = ax.legend(handles=fam_handles, loc="upper left", fontsize=8.5,
                     title="Family", title_fontsize=9, framealpha=0.95)
    ax.add_artist(leg1)
    size_handles = [Line2D([0], [0], marker="o", ls="", color="#999", mec="white",
                    ms=np.sqrt(40 + v * 9000), label=f"{v:g}")
                    for v in [0.005, 0.03, 0.08]]
    ax.legend(handles=size_handles, loc="center left", bbox_to_anchor=(0.0, 0.47),
              fontsize=8.5, title="Mean regret", title_fontsize=9, labelspacing=1.3,
              framealpha=0.95, borderpad=0.9)
    fig.tight_layout(); fig.savefig(C.FIGDIR / "fig_risk.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[step6] fig_risk.png")


def fig_showcase():
    """Paper Figure 3: 6 representative learning curves, one per message.
    Skips any panel whose context cell is absent from the data."""
    from matplotlib.lines import Line2D
    df = pd.read_parquet(C.CACHE / "long.parquet")
    sat = pd.read_parquet(C.CACHE / "saturation.parquet")
    STYLE = {"margin": ("#1f77b4", "-", 2.0, "Margin (uncertainty)"),
             "dbal": ("#d62728", "-", 1.7, "DBAL"),
             "rank2022": ("#ff7f0e", "-", 1.7, "Rank-2022"),
             "V58": ("#2ca02c", "-", 1.7, "Router (ours)"),
             "random": ("#7f7f7f", "--", 1.5, "Random"),
             "probcover": ("#9467bd", ":", 1.7, "ProbCover"),
             "robust_qbc": ("#8c564b", ":", 1.7, "Robust QBC"),
             "typiclust": ("#e377c2", ":", 1.7, "TypiClust")}
    # (regime, clf, dataset, title, message, extra strategies to emphasize)
    CASES = [
        ("tabular", "rf", "covertype_original", "Tabular / RF / covertype",
         "Diversity helps: clustering clearly above margin", ["typiclust"]),
        ("tabular", "rf", "letter_recognition_original", "Tabular / RF / letter_recognition",
         "Many classes: clustering pulls ahead, no saturation", ["typiclust"]),
        ("latent", "rf", "cifar10trainpca100", "Embeddings / RF / CIFAR-10 PCA-100",
         "Robust QBC collapses; ProbCover far behind", ["probcover", "robust_qbc"]),
        ("latent", "rf", "minsttrainpca100", "Embeddings / RF / MNIST PCA-100",
         "Robust QBC falls below its starting point", ["probcover", "robust_qbc"]),
        ("synthetic", "lr", "noisy_medium", "Synthetic / LR / noisy_medium",
         "Flat top: clustering == margin (within noise)", []),
        ("tabular", "lr", "rice_original", "Tabular / LR / rice",
         "Saturated: AL barely beats random", []),
    ]
    present = set(map(tuple, df[["regime", "clf", "dataset"]].drop_duplicates().values))
    present_ds = {d for _, _, d in present}
    CASES = [(r, c, _resolve(d, present_ds), *rest) for r, c, d, *rest in CASES]

    def panel(ax, regime, clf, ds, title, msg, emph):
        if (regime, clf, ds) not in present:
            ax.set_visible(False)
            return
        g = df[(df.regime == regime) & (df.clf == clf) & (df.dataset == ds)]
        for st in ["margin", "dbal", "rank2022", "V58", "random"] + emph:
            if st not in STYLE or st not in set(g.strategy):
                continue
            s = g[g.strategy == st].groupby("n_labeled")[C.PRIMARY_METRIC].agg(["mean", "std"])
            if s.empty:
                continue
            col, ls, lw, _ = STYLE[st]
            ax.plot(s.index, s["mean"], color=col, ls=ls, lw=lw)
            if st in ("margin",) + tuple(emph):
                ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"],
                                color=col, alpha=0.08)
        row = sat[(sat.regime == regime) & (sat.clf == clf) & (sat.dataset == ds)]
        if len(row) and row.n95.values[0]:
            ax.axvspan(g.n_labeled.min(), row.n95.values[0], color="gold", alpha=0.08)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=22)
        ax.annotate(msg, xy=(0.5, 1.02), xycoords="axes fraction", ha="center",
                    va="bottom", fontsize=10, style="italic", color="#444")
        ax.set_xlabel("Number of labeled points", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=9)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.5))
    for ax, cs in zip(axes.flat, CASES):
        panel(ax, *cs)
    axes.flat[0].set_ylabel("macro-F1", fontsize=11)
    axes.flat[3].set_ylabel("macro-F1", fontsize=11)
    handles = [Line2D([0], [0], color=STYLE[s][0], ls=STYLE[s][1], lw=2, label=STYLE[s][3])
               for s in ["margin", "dbal", "rank2022", "V58", "random",
                         "probcover", "robust_qbc", "typiclust"] if s in set(df.strategy)]
    fig.legend(handles=handles, loc="upper center", ncol=8, fontsize=10,
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=3.0)
    fig.savefig(C.FIGDIR / "fig_showcase.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("[step6] fig_showcase.png")


def run():
    fig_spread()
    fig_curves()
    fig_structure()
    fig_showcase()
    fig_risk()


if __name__ == "__main__":
    run()
