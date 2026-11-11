import sys,glob,os,numpy as np
sys.path.insert(0,'python')
import fit_bmw_i3 as F
from scipy.optimize import least_squares
src='data/raw/bmw_i3/measurement'
files=[f for f in sorted(glob.glob(src+'/Trip*.csv')) if os.path.basename(f)[:-4] not in F.EXCLUDE]
X={os.path.basename(f)[:-4]:F.load(f) for f in files}; names=list(X)
rng=np.random.default_rng(42); perm=rng.permutation(len(names)); ntr=int(.7*len(names))
tr=[names[i] for i in perm[:ntr]]; te=[names[i] for i in perm[ntr:]]
sub=lambda x:{k:(v[::5] if isinstance(v,np.ndarray) else v) for k,v in x.items()}
def run(tag,mass=None,nograde=False,fitmass=False):
    def m(p,x):
        M=p[8] if fitmass else (mass or F.MASS)
        gr=0*x['gr'] if nograde else x['gr']
        Pw=(M*x['a']+M*F.G*p[1]+M*F.G*gr+0.5*F.RHO*p[0]*x['v']**2)*x['v']/1000
        return np.where(Pw>=0,p[2]*Pw+p[3]*Pw**2,p[4]*Pw+p[5]*Pw**2)*1000+p[6]+p[7]*x['pc']
    def res(p,xs): return np.concatenate([(m(p,x)-x['Pb'])/1000 for x in xs])
    p0=[0.65,0.010,1.1,0.002,0.7,0.002,400,0.5]+([1300.] if fitmass else [])
    lo=[0.3,0.003,0.9,0,0.2,0,0,0]+([1100.] if fitmass else []); hi=[1.2,0.03,2.0,0.05,1.5,0.05,3000,3]+([1800.] if fitmass else [])
    r=least_squares(res,p0,bounds=(lo,hi),args=([sub(X[n]) for n in tr],))
    out=[]
    for grp in (tr,te):
        e=[];
        for n in grp:
            x=X[n];em=F.energy(m(r.x,x),x['t']);e0=F.energy(x['Pb'],x['t']);e.append(100*(em-e0)/e0)
        e=np.array(e); out.append('med|e| %.1f%% pooled-mean %.1f%% worst %.0f%%'%(np.median(abs(e)),e.mean(),abs(e).max()))
    print(tag,'| train',out[0],'| test',out[1],'| p',np.round(r.x,3))
run('base(1270kg)')
run('nograde',nograde=True)
run('free mass',fitmass=True)
run('free mass, nograde',fitmass=True,nograde=True)
