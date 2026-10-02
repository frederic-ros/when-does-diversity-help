"""Analysis of the replay runner output (alframework.bench.replay).

Tests the two label-free implications of the allocation model (paper Sec. 4.4):
 (i)  upside   : gain of integrated clustering over margin vs D_hat (round-0
                 allocation imbalance of the margin batch);
 (ii) downside : regret vs q_hat (fraction of the batch outside the uncertain
                 band, averaged over the rounds of the early-budget window).

AULC, n95 window, regret and failure are recomputed from the run records exactly
as in the paper (17-strategy main panel, macro-F1).

Usage:
  python tools/analyze_model_tests.py --records <records dir> --replay replay.parquet \
      [--out model_tests.txt]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon

PANEL = ["random", "margin", "entropy", "least_confident", "coreset_greedy",
         "coreset_kmeanspp", "typiclust", "probcover", "badge",
         "unc_feature_kmeans", "diversity_opt_batch", "qbc", "robust_qbc",
         "tri_committee", "adaptive_disagreement", "dbal", "rank2022"]
FAMILY = {"margin": "unc", "entropy": "unc", "least_confident": "unc",
          "coreset_greedy": "div", "coreset_kmeanspp": "div", "typiclust": "div", "probcover": "div",
          "badge": "hyb", "unc_feature_kmeans": "hyb", "diversity_opt_batch": "hyb",
          "qbc": "com", "robust_qbc": "com", "tri_committee": "com", "adaptive_disagreement": "com",
          "dbal": "clu", "rank2022": "clu", "random": "rnd"}
PUREDIV = ["coreset_greedy", "coreset_kmeanspp", "typiclust", "probcover"]
KEY = ["regime", "dataset", "classifier"]


def load_aulc(records):
    rows = []
    for f in glob.glob(f"{records}/*/*/*/split*/*.json"):
        d = json.load(open(f)); k = d["key"]
        if k["strategy"] not in PANEL:
            continue
        for h in d["history"]:
            rows.append((k["regime"], k["dataset"], k["classifier"], k["split"], k["strategy"],
                         h["n_labeled"], h["f1_macro"]))
    L = pd.DataFrame(rows, columns=KEY + ["split", "strategy", "n", "f1"])
    mc = L[L.strategy == "margin"].groupby(KEY + ["n"]).f1.mean().reset_index()
    n95 = {}
    for k, g in mc.groupby(KEY):
        g = g.sort_values("n"); first = g[g.f1 >= 0.95 * g.f1.max()].n.iloc[0]
        n95[k] = first if first > g.n.min() else g.n.max()
    out = []
    for k, g in L.groupby(KEY + ["strategy", "split"]):
        gg = g[g.n <= n95[k[:3]]].sort_values("n")
        x, y = gg.n.values, gg.f1.values
        out.append((*k, np.trapezoid(y, x) / (x[-1] - x[0]) if len(x) > 1 else y[-1]))
    return pd.DataFrame(out, columns=KEY + ["strategy", "split", "aulc"])


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return np.nan, np.nan
    r = spearmanr(x[ok], y[ok]); return r.statistic, r.pvalue


def wz(df, col, by="regime"):
    return df.groupby(by)[col].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", default="model_tests.txt")
    a = ap.parse_args()
    out = []; P = lambda s="": (print(s), out.append(s))

    A = load_aulc(a.records)
    R = pd.read_parquet(a.replay)
    P(f"Replay rows: {len(R)}, exact (ok): {R.ok.mean()*100:.2f}%  "
      f"(margin round-0 selection tie-consistent: {R.ok_margin_selection.dropna().mean()*100:.1f}%)")
    R = R[R.ok]
    cells_done = R[KEY].drop_duplicates()
    A = A.merge(cells_done, on=KEY)

    C = A.groupby(KEY + ["strategy"]).aulc.mean().unstack("strategy")
    REG = C.max(axis=1).values[:, None] - C
    RANK = C.rank(axis=1, ascending=False)
    FAIL = RANK >= C.shape[1] - 2
    P(f"Cells analysed: {len(C)} (of 86)\n")

    # ---------------- (i) upside: D_hat ----------------
    M0 = R[(R.strategy == "margin") & (R["round"] == 0)]
    Dc = M0.groupby(KEY)[["D_hat", "D_hat_half", "w_tau"]].mean()
    G = pd.DataFrame({"gain_clust": C[["dbal", "rank2022"]].max(axis=1) - C["margin"],
                      "gain_dbal": C["dbal"] - C["margin"],
                      "gain_rank": C["rank2022"] - C["margin"],
                      "gain_purediv": C[PUREDIV].max(axis=1) - C["margin"]})
    D = Dc.join(G, how="inner").reset_index()
    P("(i) UPSIDE: Spearman rho, per regime and pooled after within-regime standardisation")
    P(f"{'gain':>13} {'indicator':>11} | {'synthetic':>13} {'tabular':>13} {'embeddings':>13} | pooled")
    for g in G.columns:
        for ind in ["D_hat", "D_hat_half", "w_tau"]:
            cs = []
            for rg in ["synthetic", "tabular", "latent"]:
                s = D[D.regime == rg]; r, p = sp(s[ind], s[g]); cs.append(f"{r:+.2f} ({p:.2f})")
            r, p = sp(wz(D, ind), wz(D, g))
            P(f"{g:>13} {ind:>11} | {cs[0]:>13} {cs[1]:>13} {cs[2]:>13} | {r:+.2f} (p={p:.3f})")
    for rg, s in D.groupby("regime").D_hat:
        P(f"   D_hat {rg:9s}: mean {s.mean():.2f}, sd {s.std():.2f}, range [{s.min():.2f}, {s.max():.2f}]")
    # split level, within cell: does a split with a more imbalanced margin batch give a larger gain?
    As = A.pivot_table(index=KEY + ["split"], columns="strategy", values="aulc")
    gs = (As[["dbal", "rank2022"]].max(axis=1) - As["margin"]).rename("gain_clust_split")
    S = M0.set_index(KEY + ["split"])[["D_hat"]].join(gs, how="inner").reset_index()
    rhos = [sp(g.D_hat, g.gain_clust_split)[0] for _, g in S.groupby(KEY)]
    rhos = np.array([r for r in rhos if np.isfinite(r)])
    P(f"   split level, within cell: rho(D_hat, gain_clust) median {np.median(rhos):+.2f}, "
      f"mean {rhos.mean():+.2f}, >0 in {np.mean(rhos>0)*100:.0f}% of {len(rhos)} cells, "
      f"Wilcoxon p={wilcoxon(rhos).pvalue:.2g}")
    P()

    # ---------------- (ii) downside: q_hat ----------------
    Q = R[R.strategy.isin(PANEL)].groupby(KEY + ["strategy"])[["q_tau", "q_20"]].mean()
    Lg = Q.join(REG.stack().rename("regret")).join(FAIL.stack().rename("fail")).reset_index()
    Lg["family"] = Lg.strategy.map(FAMILY)
    Lg["cell"] = Lg.regime + "|" + Lg.dataset + "|" + Lg.classifier
    P("(ii) DOWNSIDE: mean q_hat over the early-budget window, per strategy")
    T = Lg.groupby("strategy").agg(q_tau=("q_tau", "mean"), q_20=("q_20", "mean"),
                                   fail=("fail", "mean"), regret=("regret", "mean"),
                                   r90=("regret", lambda v: v.quantile(.9))).sort_values("q_tau")
    P(T.round(3).to_string())
    P()
    for q in ["q_tau", "q_20"]:
        rh = np.array([sp(g[q], g.regret)[0] for _, g in Lg.groupby("cell")]); rh = rh[np.isfinite(rh)]
        P(f"   {q}: within-cell rho(q, regret) across 17 strategies: median {np.median(rh):+.2f}, "
          f">0 in {np.mean(rh>0)*100:.0f}% of {len(rh)} cells, Wilcoxon p={wilcoxon(rh).pvalue:.2g}")
        for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
            s = Lg[(Lg[q] >= lo) & (Lg[q] < hi)]
            if len(s):
                P(f"      {q} in [{lo:.2f},{min(hi,1):.2f}): n={len(s):4d}  fail={s.fail.mean()*100:5.1f}%  "
                  f"mean regret={s.regret.mean():.3f}  R90={s.regret.quantile(.9):.3f}")
    P()
    P("   Dose-response within a strategy, across cells (within-regime z): rho(q_tau, regret)")
    for st in PANEL:
        s = Lg[Lg.strategy == st].copy()
        if s.q_tau.std() == 0:
            continue
        r, p = sp(wz(s, "q_tau"), wz(s, "regret"))
        P(f"      {st:22s} rho={r:+.2f} (p={p:.3f})  q range [{s.q_tau.min():.2f}, {s.q_tau.max():.2f}]")
    # split level, within (cell, strategy): regret vs margin per split
    Rs = R[R.strategy.isin(PANEL)].groupby(KEY + ["strategy", "split"]).q_tau.mean().rename("q")
    As2 = A.set_index(KEY + ["strategy", "split"]).aulc
    best = A.groupby(KEY + ["split"]).aulc.max()
    regs = (best.reindex(As2.index.droplevel("strategy")).values - As2).rename("regret_split")
    SS = pd.concat([Rs, regs], axis=1, join="inner").reset_index()
    out_rows = []
    for st in PUREDIV + ["random", "robust_qbc", "dbal", "rank2022"]:
        rr = [sp(g.q, g.regret_split)[0] for _, g in SS[SS.strategy == st].groupby(KEY)]
        rr = np.array([r for r in rr if np.isfinite(r)])
        if len(rr) > 5:
            P(f"      split level, {st:18s}: median within-cell rho(q, regret) {np.median(rr):+.2f} "
              f"(>0 in {np.mean(rr>0)*100:.0f}% of {len(rr)} cells, Wilcoxon p={wilcoxon(rr).pvalue:.2g})")
    Path(a.out).write_text("\n".join(out))
    Lg.to_parquet(Path(a.out).with_suffix(".cells.parquet"))


if __name__ == "__main__":
    main()
