"""Fit a physics vehicle+battery power model to BMW i3 (60 Ah) real trips (TUM, IEEE DataPort 10.21227/6jr9-5235).
Usage: python3 -I python/fit_bmw_i3.py <measurement_dir> <out_dir>
Sign convention: battery power P_b > 0 = discharging (dataset current is negative when discharging, so P_b = -U*I).
Model v2 (kW): P_b = c1*Pw + c2*Pw^2 (Pw>=0) | r1*Pw + r2*Pw^2 (Pw<0, regen) + a0 + a1*P_cabin
(quadratic loss => efficiency depends on load; r2>0 models regen saturation / friction-brake blending)
Fixed (ASSUMED, not measured): mass. Fitted: CdA, Crr, c1, c2, r1, r2, a0, a1.  Trip-wise 70/30 split, seed 42."""
import sys, glob, os, numpy as np, pandas as pd
from scipy.optimize import least_squares
SEED=42; RHO=1.2; G=9.81; MASS=1270.0   # ASSUMPTION: 1195 kg curb + ~75 kg driver; VERIFY vs spec/readme
EXCLUDE={'TripB04','TripB09'}            # contain charging
def col(d,p):
    for c in d.columns:
        if c.strip().startswith(p): return d[c].astype(float).values
    return np.full(len(d),np.nan)
def load(f):
    d=pd.read_csv(f,sep=';',encoding='latin-1'); t=col(d,'Time'); v=col(d,'Velocity')/3.6
    U=col(d,'Battery Voltage'); I=col(d,'Battery Current'); el=col(d,'Elevation')
    w=21  # 2.1 s smoothing for accel
    k=np.ones(w)/w; vs=np.convolve(np.pad(v,(w//2,w//2),mode='edge'),k,'valid'); a=np.gradient(vs,t)
    dist=np.concatenate([[0],np.cumsum(0.5*(v[1:]+v[:-1])*np.diff(t))])
    ew=np.convolve(np.pad(el,(75,75),mode='edge'),np.ones(151)/151,'valid')
    # grade over ~100 m: use lag in distance
    j=np.searchsorted(dist,dist+100.0).clip(max=len(dist)-1); i=np.arange(len(dist))
    dd=dist[j]-dist[i]; gr=np.where(dd>50,(ew[j]-ew[i])/np.maximum(dd,1),0.0); gr=np.clip(gr,-0.12,0.12)
    pc=np.nan_to_num(col(d,'Heating Power CAN')*1000+col(d,'AirCon Power')*1000,nan=0.0)
    soc=col(d,'SoC [%]')
    return dict(t=t,v=v,a=a,gr=gr,Pb=-U*I,pc=pc,soc=soc,dt=np.median(np.diff(t)))
def model(p,x):
    cda,crr,c1,c2,r1,r2,a0,a1=p
    Pw=(MASS*x['a']+MASS*G*crr+MASS*G*x['gr']+0.5*RHO*cda*x['v']**2)*x['v']/1000.0
    Pb=np.where(Pw>=0,c1*Pw+c2*Pw**2,r1*Pw+r2*Pw**2)*1000.0+a0+a1*x['pc']; return Pb
def res(p,xs): return np.concatenate([(model(p,x)-x['Pb'])/1000 for x in xs])
def energy(P,t): return np.trapezoid(P,t)/3.6e6
if __name__=='__main__':
    src,out=sys.argv[1],sys.argv[2]; os.makedirs(out,exist_ok=True)
    files=[f for f in sorted(glob.glob(os.path.join(src,'Trip*.csv'))) if os.path.basename(f)[:-4] not in EXCLUDE]
    X={os.path.basename(f)[:-4]:load(f) for f in files}; names=list(X)
    rng=np.random.default_rng(SEED); perm=rng.permutation(len(names)); ntr=int(0.7*len(names))
    tr=[names[i] for i in perm[:ntr]]; te=[names[i] for i in perm[ntr:]]
    sub=lambda x:{k:(v[::5] if isinstance(v,np.ndarray) else v) for k,v in x.items()}
    p0=[0.65,0.010,1.1,0.002,0.7,0.002,400,0.5]; lo=[0.3,0.004,0.9,0.0,0.2,0.0,0,0]; hi=[1.2,0.03,1.6,0.05,1.0,0.05,3000,3]
    r=least_squares(res,p0,bounds=(lo,hi),args=([sub(X[n]) for n in tr],),x_scale=[0.1,0.005,0.1,0.01,0.1,0.01,300,0.5])
    p=r.x; print('params CdA,Crr,c1,c2,r1,r2,a0[W],a1:',np.round(p,4))
    rows=[]
    for n in names:
        x=X[n]; Pm=model(p,x); em=energy(Pm,x['t']); em0=energy(x['Pb'],x['t']); km=np.trapezoid(x['v'],x['t'])/1000
        rows.append(dict(trip=n,split='train' if n in tr else 'test',km=km,e_meas=em0,e_model=em,err_pct=100*(em-em0)/em0,rmse_kw=np.sqrt(np.mean((Pm-x['Pb'])**2))/1000))
    R=pd.DataFrame(rows); R.to_csv(os.path.join(out,'bmw_i3_fit_trips.csv'),index=False)
    for s in ['train','test']:
        q=R[R.split==s]; print(s,len(q),'trips | median |err| %.1f%% | mean |err| %.1f%% | worst %.1f%% | pooled err %.2f%% | RMSE kW median %.2f'%(
            q.err_pct.abs().median(),q.err_pct.abs().mean(),q.err_pct.abs().max(),100*(q.e_model.sum()-q.e_meas.sum())/q.e_meas.sum(),q.rmse_kw.median()))
    np.save(os.path.join(out,'bmw_i3_params.npy'),p)
