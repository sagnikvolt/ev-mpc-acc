"""BMW i3 fit v3: explicit auxiliary model + two traction models. Usage: python3 python/fit_bmw_i3_v3.py <measurement_dir> <out_dir>
P_b = traction(P_trac) + P_aux ; P_aux = b0 + b1*heatCAN_W + b2*aircon_W + b3*max(0,18-Tamb)*? (see code)
 M_phys : P_trac from road-load physics (mass ASSUMED 1270 kg)   -> what the MPC will use
 M_torq : P_trac = k*T*v from MEASURED motor torque (diagnostic upper bound on achievable accuracy)
Trip-wise 70/30 split seed 42. Errors reported for total, and traction-only (measured P_b minus fitted aux)."""
import sys,glob,os,numpy as np,pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import fit_bmw_i3 as F
def loadx(f):
    x=F.load(f); d=pd.read_csv(f,sep=';',encoding='latin-1')
    x['T']=np.nan_to_num(F.col(d,'Motor Torque')); amb=F.col(d,'Ambient Temperature ['); x['amb']=np.nan_to_num(amb,nan=np.nanmean(amb) if np.isfinite(np.nanmean(amb)) else 15.)
    x['heat']=np.nan_to_num(F.col(d,'Heating Power CAN'))*1000; x['ac']=np.nan_to_num(F.col(d,'AirCon Power'))*1000
    return x
def aux(b,x): return b[0]+b[1]*x['heat']+b[2]*x['ac']+b[3]*np.maximum(0,15-x['amb'])
def trac(c,P):  # P in kW
    return np.where(P>=0,c[0]*P+c[1]*P**2,c[2]*P+c[3]*P**2)*1000
def Pphys(q,x):
    Pw=(F.MASS*x['a']+F.MASS*F.G*q[0]+F.MASS*F.G*x['gr']+0.5*F.RHO*q[1]*x['v']**2)*x['v']/1000; return Pw
def Ptorq(q,x): return q[0]*x['T']*x['v']/1000
def make(kind):
    if kind=='phys': nq=2;q0=[0.01,0.5];qlo=[0.003,0.2];qhi=[0.03,1.2];Pf=Pphys
    else: nq=1;q0=[0.5];qlo=[0.05];qhi=[5.0];Pf=Ptorq
    def pred(p,x):
        q=p[:nq];c=p[nq:nq+4];b=p[nq+4:]; return trac(c,Pf(q,x))+aux(b,x)
    p0=q0+[1.1,0.002,0.8,0.002]+[500,0.5,0.5,10]; lo=qlo+[0.8,0,0.2,0]+[0,0,0,0]; hi=qhi+[2.0,0.05,1.5,0.05]+[3000,3,3,300]
    return pred,p0,lo,hi,nq
if __name__=='__main__':
    src,out=sys.argv[1:3]; os.makedirs(out,exist_ok=True)
    files=[f for f in sorted(glob.glob(src+'/Trip*.csv')) if os.path.basename(f)[:-4] not in F.EXCLUDE]
    X={os.path.basename(f)[:-4]:loadx(f) for f in files}; names=list(X)
    rng=np.random.default_rng(42);perm=rng.permutation(len(names));ntr=int(.7*len(names))
    tr=[names[i] for i in perm[:ntr]];te=[names[i] for i in perm[ntr:]]
    sub=lambda x:{k:(v[::5] if isinstance(v,np.ndarray) else v) for k,v in x.items()}
    allrows=[]
    for kind in ['phys','torq']:
        pred,p0,lo,hi,nq=make(kind)
        r=least_squares(lambda p:np.concatenate([(pred(p,sub(X[n]))-sub(X[n])['Pb'])/1000 for n in tr]),p0,bounds=(lo,hi),x_scale=np.maximum(np.abs(p0),1e-2))
        p=r.x; print(kind,'params',np.round(p,4))
        for n in names:
            x=X[n];Pm=pred(p,x);ea=F.energy(aux(p[nq+4:],x),x['t']);e0=F.energy(x['Pb'],x['t']);em=F.energy(Pm,x['t'])
            allrows.append(dict(model=kind,trip=n,split='train' if n in tr else 'test',e_meas=e0,e_model=em,err_total=100*(em-e0)/e0,
              err_trac=100*((em-ea)-(e0-ea))/(e0-ea),aux_kwh=ea,rmse_kw=np.sqrt(np.mean((Pm-x['Pb'])**2))/1000))
        np.save(os.path.join(out,'bmw_i3_params_v3_%s.npy'%kind),p)
    R=pd.DataFrame(allrows); R.to_csv(os.path.join(out,'bmw_i3_fit_trips_v3.csv'),index=False)
    for m in ['phys','torq']:
        for s in ['train','test']:
            q=R[(R.model==m)&(R.split==s)]
            print(m,s,len(q),'| total: med|e| %.1f%% mean(signed) %.1f%% max %.0f%% | traction-only: med|e| %.1f%% max %.0f%% | pooled total %.1f%% | RMSE %.2f kW'%(
              q.err_total.abs().median(),q.err_total.mean(),q.err_total.abs().max(),q.err_trac.abs().median(),q.err_trac.abs().max(),
              100*(q.e_model.sum()-q.e_meas.sum())/q.e_meas.sum(),q.rmse_kw.median()))
