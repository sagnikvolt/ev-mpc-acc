"""Fit the final physics+aux model on ALL BMW i3 trips (model form of cv_bmw_i3_final.py) and freeze parameters for the MPC.
Out: results/bmw_i3_params_final.npy  p=[Crr,c1,c2,r1,r2,b0,b1,b2,b3,CdA]"""
import sys,glob,os,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import fit_bmw_i3 as F, fit_bmw_i3_v3 as V
src=os.path.expanduser('~/mnt/EV Project/data/raw/bmw_i3/measurement')
files=[f for f in sorted(glob.glob(src+'/Trip*.csv')) if os.path.basename(f)[:-4] not in F.EXCLUDE]
X=[V.loadx(f) for f in files]; sub=lambda x:{k:(v[::5] if isinstance(v,np.ndarray) else v) for k,v in x.items()}
def pred(p,x):
    Pw=(F.MASS*x['a']+F.MASS*F.G*p[0]+F.MASS*F.G*x['gr']+0.5*F.RHO*p[9]*x['v']**2)*x['v']/1000
    return V.trac(p[1:5],Pw)+V.aux(p[5:9],x)
p0=[0.009,1.1,0.002,0.7,0.002,500,0.5,0.5,10,0.6]; lo=[0.004,1.0,0,0.2,0,0,0,0,0,0.3]; hi=[0.02,1.6,0.05,1.0,0.05,3000,3,3,300,1.2]
S=[sub(x) for x in X]
r=least_squares(lambda p:np.concatenate([(pred(p,s)-s['Pb'])/1000 for s in S]),p0,bounds=(lo,hi),x_scale=np.maximum(np.abs(p0),1e-2))
np.save(os.path.expanduser('~/mnt/EV Project/results/bmw_i3_params_final.npy'),r.x); print(len(files),'trips; params',np.round(r.x,4))
