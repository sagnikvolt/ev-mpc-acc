"""Gate 2 check (Opus): 5-fold trip-wise CV with PHYSICALLY CONSTRAINED parameters.
Mass 1270 kg (BMW press kit 07/2013, EU incl. 75 kg driver), CdA 0.69 m2 (Cd 0.29 x A 2.38, same source) FIXED.
Regen coefficient r1 <= 1 (cannot recover more than wheel power). Fitted: Crr, traction loss c1,c2, r1,r2, aux b0..b3.
Usage: python3 python/cv_bmw_i3_physical.py <measurement_dir> <out_dir>"""
import sys,glob,os,numpy as np,pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import fit_bmw_i3 as F, fit_bmw_i3_v3 as V
CDA=0.69
src,out=sys.argv[1:3]
files=[f for f in sorted(glob.glob(src+'/Trip*.csv')) if os.path.basename(f)[:-4] not in F.EXCLUDE]
X={os.path.basename(f)[:-4]:V.loadx(f) for f in files}; names=list(X)
sub=lambda x:{k:(v[::5] if isinstance(v,np.ndarray) else v) for k,v in x.items()}
def pred(p,x):
    Pw=(F.MASS*x['a']+F.MASS*F.G*p[0]+F.MASS*F.G*x['gr']+0.5*F.RHO*CDA*x['v']**2)*x['v']/1000
    return V.trac(p[1:5],Pw)+V.aux(p[5:],x)
p0=[0.009,1.1,0.002,0.7,0.002,500,0.5,0.5,10]; lo=[0.004,1.0,0,0.2,0,0,0,0,0]; hi=[0.02,1.6,0.05,1.0,0.05,3000,3,3,300]
perm=np.random.default_rng(42).permutation(len(names)); rows=[]; P=[]
for k,fi in enumerate(np.array_split(perm,5)):
    te=[names[i] for i in fi]; tr=[n for n in names if n not in te]
    r=least_squares(lambda p:np.concatenate([(pred(p,sub(X[n]))-sub(X[n])['Pb'])/1000 for n in tr]),p0,bounds=(lo,hi),x_scale=np.maximum(np.abs(p0),1e-2)); P.append(r.x)
    for n in te:
        x=X[n]; e0=F.energy(x['Pb'],x['t']); em=F.energy(pred(r.x,x),x['t'])
        rows.append(dict(trip=n,cat=n[4],km=np.trapezoid(x['v'],x['t'])/1000,e_meas=e0,e_model=em,err=100*(em-e0)/e0))
R=pd.DataFrame(rows); R.to_csv(os.path.join(out,'bmw_i3_cv_trips_physical.csv'),index=False)
P=np.array(P); print('params mean over folds [Crr,c1,c2,r1,r2,b0,b1,b2,b3]:',np.round(P.mean(0),4)); print('at bound count per param:',[int(((abs(P[:,i]-lo[i])<1e-6)|(abs(P[:,i]-hi[i])<1e-6)).sum()) for i in range(len(lo))])
a=R.err.abs(); print('CV %d trips: median|err| %.1f%%  mean %.1f%%  p90 %.1f%%  pooled %.2f%%  A %.1f%%  B %.1f%%  within5 %.0f%%'%(len(R),a.median(),a.mean(),a.quantile(.9),100*(R.e_model.sum()-R.e_meas.sum())/R.e_meas.sum(),R[R.cat=='A'].err.abs().median(),R[R.cat=='B'].err.abs().median(),100*(a<5).mean()))
R['whkm_err']=(R.e_model-R.e_meas)/R.km*1000; print('Wh/km MAE %.1f (median measured %.0f Wh/km)'%(R.whkm_err.abs().mean(),(R.e_meas/R.km*1000).median()))
