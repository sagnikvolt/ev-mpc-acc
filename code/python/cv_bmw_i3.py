"""5-fold trip-wise cross-validation of the physics+aux BMW i3 model (v3 'phys'). Usage: python3 python/cv_bmw_i3.py <measurement_dir> <out_dir>"""
import sys,glob,os,numpy as np,pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import fit_bmw_i3 as F, fit_bmw_i3_v3 as V
src,out=sys.argv[1:3]
files=[f for f in sorted(glob.glob(src+'/Trip*.csv')) if os.path.basename(f)[:-4] not in F.EXCLUDE]
X={os.path.basename(f)[:-4]:V.loadx(f) for f in files}; names=list(X)
sub=lambda x:{k:(v[::5] if isinstance(v,np.ndarray) else v) for k,v in x.items()}
pred,p0,lo,hi,nq=V.make('phys'); rng=np.random.default_rng(42); perm=rng.permutation(len(names)); folds=np.array_split(perm,5); rows=[]
for k,fi in enumerate(folds):
    te=[names[i] for i in fi]; tr=[n for n in names if n not in te]
    r=least_squares(lambda p:np.concatenate([(pred(p,sub(X[n]))-sub(X[n])['Pb'])/1000 for n in tr]),p0,bounds=(lo,hi),x_scale=np.maximum(np.abs(p0),1e-2))
    for n in te:
        x=X[n]; e0=F.energy(x['Pb'],x['t']); em=F.energy(pred(r.x,x),x['t']); km=np.trapezoid(x['v'],x['t'])/1000
        rows.append(dict(fold=k,trip=n,cat=n[4],km=km,amb=x['amb'].mean(),vmean=x['v'].mean()*3.6,e_meas=e0,e_model=em,err=100*(em-e0)/e0,rmse_kw=np.sqrt(np.mean((pred(r.x,x)-x['Pb'])**2))/1000))
R=pd.DataFrame(rows); R.to_csv(os.path.join(out,'bmw_i3_cv_trips.csv'),index=False)
a=R.err.abs(); print('CV over',len(R),'trips: median|err| %.1f%%, mean|err| %.1f%%, 90th pct %.1f%%, share within 5%%: %.0f%%, within 10%%: %.0f%%'%(a.median(),a.mean(),a.quantile(.9),100*(a<5).mean(),100*(a<10).mean()))
print('pooled energy error %.2f%% | signed mean %.1f%%'%(100*(R.e_model.sum()-R.e_meas.sum())/R.e_meas.sum(),R.err.mean()))
print(R.groupby('cat').apply(lambda q:pd.Series(dict(n=len(q),med_abs=q.err.abs().median(),signed=q.err.mean())),include_groups=False).round(1))
R['kmbin']=pd.cut(R.km,[0,8,15,25,70]); print(R.groupby('kmbin',observed=True).apply(lambda q:pd.Series(dict(n=len(q),med_abs=q.err.abs().median())),include_groups=False).round(1))
print('worst 6:'); print(R.reindex(R.err.abs().sort_values(ascending=False).index).head(6)[['trip','km','amb','vmean','e_meas','err']].round(1).to_string(index=False))
