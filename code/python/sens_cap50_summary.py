import glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
R=os.path.expanduser('~/mnt/EV Project'); G=f'{R}/results/gate3b'; O=f'{R}/results/sens_cap50'
S=np.load(f'{R}/cache/openacc_segs.npy',allow_pickle=True); W=pd.read_csv(f'{R}/results/windows.csv')
def rd(files): return pd.concat([pd.read_csv(f) for f in files]).drop_duplicates('win').set_index('win')
def lead_rms(w):
    r=W.iloc[w]; x=S[int(r.seg)]['vl'][int(r.w0):int(r.w0)+1200]; return np.sqrt(np.mean((np.diff(x)/0.1)**2))
def comp(A,B,label):
    ix=A.index.intersection(B.index); d=(B.loc[ix,'whkm_adj']/A.loc[ix,'whkm_adj']-1)*100
    rms=pd.Series({w:lead_rms(w) for w in ix}); t=pd.cut(rms,[-1,0.278,0.414,99],labels=['calm','medium','dynamic'])
    print(label,'n',len(ix),'median %.2f mean %.2f p %.1e'%(d.median(),d.mean(),wilcoxon(d).pvalue), {k:round(d[t==k].median(),2) for k in ['calm','medium','dynamic']}, 'unsafe B',int((B.loc[ix,'short_max']>0.5).sum()) if 'short_max' in B else '')
print('-- cap 35 (existing) --')
a35=rd([f'{G}/r0_0.65_acc_0.csv']); o35=rd(sorted(glob.glob(f'{G}/r0_0.65_oracle_s1_n27_*.csv')))
comp(a35,o35,'oracle vs ACC cap35')
print('-- cap 50 --')
a50=rd([f'{O}/acc.csv']); o50=rd(sorted(glob.glob(f'{O}/oracle_[0-4].csv')))
comp(a50,o50,'oracle vs ACC cap50')
print('ACC energy cap35 vs cap50 (mean Wh/km):',a35.whkm_adj.mean().round(2),a50.whkm_adj.mean().round(2),'| oracle',o35.whkm_adj.mean().round(2),o50.whkm_adj.mean().round(2))
print('-- S1b --')
def L(a): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{G}/s1b_test_{a}_*.csv'))]).drop_duplicates('win').set_index('win')
def cmp2(A,B,label):
    d=(B.whkm_adj/A.whkm_adj-1)*100; print(label,'n',len(d),'median %.2f better %d p %.1e'%(d.median(),(d<0).sum(),wilcoxon(d).pvalue),'unsafe',int((B.short_max>0.5).sum()),'maxshort %.2f'%B.short_max.max(),'cross_excess mean %.2f'%B.cross_excess.mean(),'| ACC-map excess %.2f'%A.cross_excess.mean())
cmp2(L('acc_map'),L('mpc_map_L'),'cap35 MPC-map-L vs ACC-map')
A50=pd.read_csv(f'{O}/s1b_acc.csv').set_index('win'); M50=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{O}/s1b_L_[0-2].csv'))]).set_index('win')
cmp2(A50,M50,'cap50 MPC-map-L vs ACC-map')
