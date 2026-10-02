"""Robustness grid for the allocation-imbalance proxy D_hat (paper Table D-robust).
Usage: python tools/analyze_imbalance.py <records dir> <replay_imbalance.parquet> [out.csv]
Also writes <out>.tex, the body of the paper table on the imbalance proxy.
Input: output of `python -m alframework.bench.replay_imbalance`.
"""
import sys, pandas as pd, numpy as np
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_model_tests import load_aulc
from scipy.stats import spearmanr, wilcoxon
K=['regime','dataset','classifier']
I=pd.read_parquet(sys.argv[2])
print('rows',len(I),'ok',round(I.ok.mean(),4)); I=I[I.ok]
print('degenerate share by region/M:'); print(I.groupby(['region','M']).degenerate.mean().round(3).unstack())
A=load_aulc(sys.argv[1])
As=A.pivot_table(index=K+['split'],columns='strategy',values='aulc')
PD=['coreset_greedy','coreset_kmeanspp','typiclust','probcover']
gs=pd.DataFrame({'clust':As[['dbal','rank2022']].max(axis=1)-As.margin,'dbal':As.dbal-As.margin,'rank':As.rank2022-As.margin,'purediv':As[PD].max(axis=1)-As.margin})
gc=gs.groupby(K).mean()
def sp(x,y):
    r=spearmanr(x,y); return r.statistic,r.pvalue
def wz(df,c): return df.groupby('regime')[c].transform(lambda v:(v-v.mean())/(v.std()+1e-12))
rows=[]
for when in ['round0','window']:
    J=I[I['round']==0] if when=='round0' else I
    Ds=J.groupby(K+['split','region','M']).D_hat.mean().rename('D').reset_index()
    for (reg,M),d in Ds.groupby(['region','M']):
        dc=d.groupby(K).D.mean().to_frame().join(gc,how='inner').reset_index()
        ds=d.set_index(K+['split']).join(gs,how='inner').reset_index()
        for g in ['clust','dbal','rank','purediv']:
            r,p=sp(wz(dc,'D'),wz(dc,g))
            per=[sp(dc[dc.regime==x].D,dc[dc.regime==x][g])[0] for x in ['synthetic','tabular','latent']]
            rr=np.array([spearmanr(h.D,h[g]).statistic for _,h in ds.groupby(K) if h.D.std()>0])
            rr=rr[np.isfinite(rr)]
            rows.append(dict(when=when,region=reg,M=M,gain=g,rho_cell=r,p_cell=p,syn=per[0],tab=per[1],emb=per[2],
                             rho_split_med=np.median(rr),pos_share=(rr>0).mean(),p_split=wilcoxon(rr).pvalue,n_cells=len(rr)))
T=pd.DataFrame(rows)
pd.set_option('display.width',200)
print(T.round(3).to_string())
T.to_csv(sys.argv[3] if len(sys.argv)>3 else 'D_robustness.csv',index=False)
print('\nAll cell-level p (clust):',T[T.gain=='clust'].p_cell.round(3).tolist())
print('min p over 48 tests:',T.p_cell.min().round(4), T.p_split.min().round(4))
d=I[(I['round']==0)&(I.region=='U01')]
print('U01 region size at round0 (median by regime):',d.groupby('regime').region_size.median().to_dict())

# ---- LaTeX body of the paper table (gain of integrated clustering) ----
c = T[T.gain == "clust"].copy()
c["o"] = c.when.map({"round0": 0, "window": 1}) * 100 + c.region.map({"U01": 0, "top20": 1}) * 10 + c.M.map({"half": 0, "k": 1, "2k": 2})
lab = {"round0": "First", "window": "Window"}
rl = {"U01": r"$\mathcal{U}_{0.1}$", "top20": r"Lowest 20\%"}
ml = {"half": "$k/2$", "k": "$k$", "2k": "$2k$"}
lines = [f"{lab[r.when]} & {rl[r.region]} & {ml[r.M]} & ${r.rho_cell:+.2f}$ & {r.p_cell:.3f} & "
         f"${r.rho_split_med:+.2f}$ & {r.p_split:.3f} \\\\" for r in c.sort_values("o").itertuples()]
out_csv = sys.argv[3] if len(sys.argv) > 3 else "D_robustness.csv"
open(os.path.splitext(out_csv)[0] + ".tex", "w").write("\n".join(lines) + "\n")
