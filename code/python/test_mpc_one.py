import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpc_acc as A
S = np.load(os.path.expanduser('~/mnt/EV Project/cache/openacc_segs.npy'), allow_pickle=True)
c = [s for s in S if s['pair'] == 0 and s['lead_acc'] < 0.5 and len(s['vl']) >= 1200 and s['vl'].mean() > 8]
print(len(c), 'candidate segments'); c.sort(key=lambda z: -z['vl'][:1200].std()); s = c[int(sys.argv[1]) if len(sys.argv)>1 else 0]; vl = s['vl'][:1200]; print(s['campaign'], s['file'], 'vmean', vl.mean().round(1), 'vstd', vl.std().round(2))
for pv in ['const', 'oracle']:
    m = A.MPC(); t = time.time(); r = A.simulate(vl, m, preview=pv); print(pv, round(time.time() - t, 1), 's', {k: round(float(x), 3) for k, x in r.items()})
