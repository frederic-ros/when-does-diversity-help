"""
step7_tables.py — generate LaTeX table fragments used by paper.tex.

  tab1_leaderboard.tex      : global leaderboard (18 strategies)
  tab2_perregime.tex        : mean rank per regime
  tab3_collapse.tex         : collapse rate per regime
  tab6_struct_purediv.tex   : pure-diversity gain vs structure (Spearman)
  tab6b_struct_clustering.tex: clustering-family gain vs structure (Spearman)

Inputs:  out/cache/aulc_split.parquet, global_summary.parquet, delta_structure.parquet
Outputs: out/tables/*.tex
Run standalone:  python step7_tables.py
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C

WIN = "aulc_adaptive"


PAPER_NAME = {"unc_feature_kmeans": "Unc.-aug.\\ $k$-means",
              "diversity_optimized_batch": "Div.-opt.\\ batch",
              "adaptive_disagreement": "Adapt.\\ disagreement",
              "coreset_kmeanspp": "Core-set ($k$-means++)"}


def _name(s):
    return PAPER_NAME.get(s, C.PRETTY[s])


def _f(x, d=2):
    return f"{x:.{d}f}"


def tab1_leaderboard():
    """Paper Table 2: rank (F1, BA), win, regret mean / R90 / max, failure rate."""
    G = pd.read_parquet(C.CACHE / "global_summary.parquet").copy()
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    cell = (S[S.metric == "f1_macro"].groupby(["regime", "clf", "dataset", "strategy"])[WIN]
            .mean().unstack("strategy"))
    regmax = (cell.max(axis=1).values[:, None] - cell).max()
    G["max_f1"] = G["strategy"].map(regmax)
    G["fam"] = G["family"].map(C.FAM_SHORT)
    G = G.sort_values("rank_f1")
    rows = [
        f"{_name(r.strategy)} & {r.fam} & {_f(r.rank_f1)} & {_f(r.rank_ba)} & "
        f"{_f(r.win_f1*100,0)}\\% & {_f(r.reg_f1,3)} & {_f(r.p90_f1,3)} & "
        f"{_f(r.max_f1,3)} & {_f(r.coll_f1*100,0)}\\% \\\\"
        for r in G.itertuples()
    ]
    (C.TABDIR / "tab1_leaderboard.tex").write_text("\n".join(rows))


def _cellrank(S, metric):
    c = (S[S.metric == metric]
         .groupby(["regime", "clf", "dataset", "strategy"])[WIN].mean().reset_index())
    grp = c.groupby(["regime", "clf", "dataset"])
    c["rank"] = grp[WIN].rank(ascending=False)
    c["worst3"] = grp[WIN].rank(ascending=True) <= 3
    return c


def tab2_perregime():
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    c = _cellrank(S, "f1_macro")
    piv = c.groupby(["strategy", "regime"])["rank"].mean().unstack("regime")
    piv["overall"] = c.groupby("strategy")["rank"].mean()   # pooled over the 86 settings
    piv = piv.sort_values("overall")
    piv.index = [_name(i) for i in piv.index]
    rows = [f"{i} & {_f(r.synthetic)} & {_f(r.tabular)} & {_f(r.latent)} & {_f(r.overall)} \\\\"
            for i, r in piv.iterrows()]
    (C.TABDIR / "tab2_perregime.tex").write_text("\n".join(rows))


def tab3_collapse():
    S = pd.read_parquet(C.CACHE / "aulc_split.parquet")
    c = _cellrank(S, "f1_macro")
    cr = c.groupby(["strategy", "regime"])["worst3"].mean().unstack("regime")
    cr["overall"] = c.groupby("strategy")["worst3"].mean()  # pooled over the 86 settings
    cr = cr.sort_values("overall")
    cr.index = [_name(i) for i in cr.index]
    rows = [f"{i} & {_f(r.synthetic*100,0)}\\% & {_f(r.tabular*100,0)}\\% & "
            f"{_f(r.latent*100,0)}\\% & {_f(r.overall*100,0)}\\% \\\\"
            for i, r in cr.iterrows()]
    (C.TABDIR / "tab3_collapse.tex").write_text("\n".join(rows))


def _sp_star(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return "--"
    r, p = spearmanr(x, y)
    star = "" if p >= 0.05 else ("$^{*}$" if p >= 0.01 else "$^{**}$")
    return f"${r:+.2f}${star}"


def _wz(M, col):
    return M.groupby("regime")[col].transform(lambda v: (v - v.mean()) / (v.std() + 1e-12))


def _struct_row(M, key, target, lbl):
    cells = [_sp_star(M[M.regime == r][key], M[M.regime == r][target])
             for r in ["synthetic", "tabular", "latent"]]
    pooled = _sp_star(_wz(M, key), _wz(M, target))
    return f"{lbl} & " + " & ".join(cells) + f" & {pooled} \\\\"


def _struct_table(target, fname):
    """Per-regime Spearman correlations; pooled column after within-regime
    standardisation (as in the paper)."""
    fp = C.CACHE / "delta_structure.parquet"
    if not fp.exists():
        print(f"[step7] skipping {fname} (run step 5 first)")
        return
    M = pd.read_parquet(fp)
    feats = [("low_margin_frac", "Uncertain fraction $w_{0.1}$"), ("mean_margin", "Mean margin"),
             ("eff_dim_unc", "Eff.\\ dimension$^{\\dagger}$"),
             ("clusterability", "Clusterability$^{\\dagger}$"), ("n_classes", "Number of classes")]
    rows = [_struct_row(M, k, target, l) for k, l in feats]
    (C.TABDIR / fname).write_text("\n".join(rows))


def tab_struct_paper():
    """Paper Table 6 body (pure-diversity block + integrated-clustering row)."""
    fp = C.CACHE / "delta_structure.parquet"
    if not fp.exists():
        return
    M = pd.read_parquet(fp)
    rows = ["\\multicolumn{5}{l}{\\emph{Gain of pure diversity}}\\\\"]
    for k, l in [("low_margin_frac", "Uncertain fraction $w_{0.1}$"),
                 ("eff_dim_unc", "Eff.\\ dimension$^{\\dagger}$"),
                 ("clusterability", "Clusterability$^{\\dagger}$"),
                 ("n_classes", "Number of classes")]:
        rows.append(_struct_row(M, k, "delta_purediv", l))
    rows += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Gain of integrated clustering}}\\\\",
             _struct_row(M, "low_margin_frac", "delta_clustering", "Uncertain fraction $w_{0.1}$")]
    (C.TABDIR / "tab6_struct_paper.tex").write_text("\n".join(rows))


def tab_cost():
    """Appendix table: median selection time per batch (ms) by strategy and regime,
    from the t_select field of the run records (single-threaded runs; includes the
    scoring of the pool by the classifier)."""
    fp = C.CACHE / "long_full.parquet"
    if not fp.exists():
        print("[step7] skipping tab_cost (needs the run records)")
        return
    d = pd.read_parquet(fp, columns=["regime", "strategy", "t_select"])
    d = d[d.t_select.notna() & ~d.strategy.map(C.norm_strat).isin(C.EXTRAS)]
    d["strategy"] = d["strategy"].map(C.norm_strat)
    T = d.groupby(["strategy", "regime"]).t_select.median().unstack("regime") * 1000
    T["all"] = d.groupby("strategy").t_select.median() * 1000
    T["ratio"] = T["all"] / T.loc["margin", "all"]
    T = T.sort_values("all")
    fmt = lambda v: f"{v:,.0f}" if v >= 10 else f"{v:.1f}"
    rows = [f"{_name(i)} & {fmt(r.synthetic)} & {fmt(r.tabular)} & {fmt(r.latent)} & "
            f"{('$<$0.1' if r.ratio < 0.05 else format(r.ratio, '.1f'))} \\\\" for i, r in T.iterrows()]
    (C.TABDIR / "tab_cost.tex").write_text("\n".join(rows))


def run():
    tab1_leaderboard()
    tab2_perregime()
    tab3_collapse()
    _struct_table("delta_purediv", "tab6_struct_purediv.tex")
    _struct_table("delta_clustering", "tab6b_struct_clustering.tex")
    tab_struct_paper()
    tab_cost()
    print(f"[step7] wrote LaTeX table fragments -> {C.TABDIR}")


if __name__ == "__main__":
    run()
