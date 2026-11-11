"""Stage 1 of ML residual benchmark. Per fold (5-fold trip-wise, seed 42, same as cv_bmw_i3_final.py):
refit physics ONLY on that fold's training trips, then compute physics prediction + residual target for ALL trips.
Data at 2 Hz (every 5th 10 Hz sample). Output: cache/ml_data.npz (+ folds). Usage: python3 python/ml_prep.py <measurement_dir>"""
import sys,glob,os,numpy as np,pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import fit_bmw_i3 as F, fit_bmw_i3_v3 as V
STEP=5
def phys_pred(p,x):
    Pw=(F.MASS*x['a']+F.MASS*F.G*p[0]+F.MASS*F.G*x['gr']+0.5*F.RHO*p[9]*x['v']**2)*x['v']/1000
    return V.trac(p[1:5],Pw)+V.aux(p[5:9],x), Pw
P0=[0.009,1.1,0.002,0.7,0.002,500,0.5,0.5,10,0.6]; LO=[0.004,1.0,0,0.2,0,0,0,0,0,0.3]; HI=[0.02,1.6,0.05,1.0,0.05,3000,3,3,300,1.2]
def extra(f):
    d=pd.read_csv(f,sep=';',encoding='latin-1'); bt=F.col(d,'Battery Temperature [')
    soc=F.col(d,'SoC [%]'); 
    return bt,soc
if __name__=='__main__':
    src=sys.argv[1]
    files=[f for f in sorted(glob.glob(src+'/Trip*.csv')) if os.path.basename(f)[:-4] not in F.EXCLUDE]
    X={};
    for f in files:
        n=os.path.basename(f)[:-4]; x=V.loadx(f); bt,soc=extra(f); x['bt']=bt; x['soc']=soc; X[n]=x
    names=list(X); perm=np.random.default_rng(42).permutation(len(names)); folds=[[names[i] for i in fi] for fi in np.array_split(perm,5)]
    sub=lambda x:{k:(v[::STEP] if isinstance(v,np.ndarray) else v) for k,v in x.items()}
    out={}; meta=[]
    for k,te in enumerate(folds):
        tr=[n for n in names if n not in te]
        r=least_squares(lambda p:np.concatenate([(phys_pred(p,sub(X[n])[0:1][0] if False else sub(X[n]))[0]-sub(X[n])['Pb'])/1000 for n in tr]),P0,bounds=(LO,HI),x_scale=np.maximum(np.abs(P0),1e-2))
        out['phys_params_%d'%k]=r.x; print('fold',k,'phys params',np.round(r.x,3),flush=True)
    # features at 2 Hz, per trip; physics outputs per fold
    rows=[]
    for n in names:
        x=sub(X[n]); N=len(x['t']); lag=lambda a,m:np.concatenate([np.full(m,a[0]),a[:-m]])
        soc=x['soc'].copy(); 
        if np.isnan(soc).all(): soc[:]=np.nan
        F_=dict(v=x['v'],a=x['a'],gr=x['gr'],v2=x['v']**2,av=x['a']*x['v'],heat=x['heat'],ac=x['ac'],amb=x['amb'],bt=x['bt'],soc=soc,
                v_l2=lag(x['v'],4),v_l5=lag(x['v'],10),a_l2=lag(x['a'],4),a_l5=lag(x['a'],10),
                v_m5=pd.Series(x['v']).rolling(10,min_periods=1).mean().values,a_m5=pd.Series(x['a']).rolling(10,min_periods=1).mean().values)
        rows.append((n,x,F_))
    feat_names=list(rows[0][2].keys()); out['feat_names']=np.array(feat_names)
    Xf=np.concatenate([np.column_stack([r[2][f] for f in feat_names]) for r in rows]); out['X']=Xf.astype(np.float32)
    out['trip']=np.concatenate([[names.index(r[0])]*len(r[1]['t']) for r in rows]); out['t']=np.concatenate([r[1]['t'] for r in rows])
    out['Pb']=np.concatenate([r[1]['Pb'] for r in rows]); out['names']=np.array(names)
    out['fold_of_trip']=np.array([[k for k,te in enumerate(folds) if n in te][0] for n in names])
    for k in range(5):
        p=out['phys_params_%d'%k]; out['Pphys_%d'%k]=np.concatenate([phys_pred(p,r[1])[0] for r in rows]); out['Pw_%d'%k]=np.concatenate([phys_pred(p,r[1])[1] for r in rows])
    np.savez_compressed('cache/ml_data.npz',**out); print('saved',Xf.shape,'features',feat_names)
