"""Extrapolation test (Opus, Gate 3). Train on one condition, test on another; physics refit on train trips only.
Splits: summer(A)->winter(B), winter->summer, low-speed (vmax<100 km/h)->high-speed. Compares physics, physics+LightGBM residual, pure LightGBM."""
import sys,os,glob,numpy as np,lightgbm as lgb
from scipy.optimize import least_squares
sys.path.insert(0,'python'); import ml_prep as M
d=np.load('cache/ml_data.npz',allow_pickle=True); X=d['X'].astype(float); Pb=d['Pb']; trip=d['trip']; tt=d['t']; names=list(d['names']); fn=list(d['feat_names'])
# rebuild per-trip raw dicts at 2 Hz for physics refit
import fit_bmw_i3 as F, fit_bmw_i3_v3 as V
raw={}
for n in names:
    x=V.loadx(f'data/raw/bmw_i3/measurement/{n}.csv'); raw[n]={k:(v[::5] if isinstance(v,np.ndarray) else v) for k,v in x.items()}
vmax={n:raw[n]['v'].max()*3.6 for n in names}
def run(tag,trn,tst):
    trn=[names.index(n) for n in trn]; tst=[names.index(n) for n in tst]
    r=least_squares(lambda p:np.concatenate([(M.phys_pred(p,raw[names[i]])[0]-raw[names[i]]['Pb'])/1000 for i in trn]),M.P0,bounds=(M.LO,M.HI),x_scale=np.maximum(np.abs(M.P0),1e-2))
    Pp=np.concatenate([M.phys_pred(r.x,raw[n])[0] for n in names]); Pw=np.concatenate([M.phys_pred(r.x,raw[n])[1] for n in names])
    tr=np.isin(trip,trn); te=np.isin(trip,tst); Xa=np.column_stack([np.nan_to_num(X,nan=0.0),Pw])
    mk=lambda:lgb.LGBMRegressor(n_estimators=400,learning_rate=0.05,num_leaves=31,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=42,n_jobs=2,verbose=-1)
    pr=Pp[te]+mk().fit(Xa[tr],(Pb-Pp)[tr]).predict(Xa[te]); pu=mk().fit(np.nan_to_num(X[tr]),Pb[tr]).predict(np.nan_to_num(X[te]))
    out=[]
    for nm,p in [('physics',Pp[te]),('phys+LGBM',pr),('pure LGBM',pu)]:
        P=np.zeros(len(Pb)); P[te]=p; errs=[]
        for i in tst:
            m=trip==i; e0=np.trapezoid(Pb[m],tt[m]); e1=np.trapezoid(P[m],tt[m]); errs.append(100*(e1-e0)/e0)
        out.append('%s rmse %.2f kW, trip med|e| %.1f%%, mean signed %+.1f%%'%(nm,np.sqrt(np.mean((p-Pb[te])**2))/1000,np.median(np.abs(errs)),np.mean(errs)))
    print(f'== {tag} (train {len(trn)}, test {len(tst)} trips)'); print('\n'.join(out),flush=True)
A=[n for n in names if n[4]=='A']; B=[n for n in names if n[4]=='B']
run('summer -> winter',A,B); run('winter -> summer',B,A)
lo=[n for n in names if vmax[n]<100]; hi=[n for n in names if vmax[n]>=100]; run('vmax<100 -> vmax>=100',lo,hi)
