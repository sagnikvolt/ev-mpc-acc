"""Opus check: stronger kinematic baseline = constant-acceleration preview with decay (tau=2 s), lead accel from 1 s finite difference.
Same windows as results/gate3b/oracle_all.csv. Usage: python3 python/gate3b_caccel.py <part> <nparts>"""
import sys, os, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpc_acc as A
ROOT = os.path.expanduser('~/mnt/EV Project'); S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True)
R = pd.read_csv(f'{ROOT}/results/gate3b/oracle_all.csv')
def fc(h, N, dt, tau=2.0):
    v0 = h[-1]; a = np.clip((h[-1] - h[-11]) / 1.0, -3.5, 2.0) if len(h) > 11 else 0.0
    t = (np.arange(N) + 0.5) * dt; return np.clip(v0 + a * tau * (1 - np.exp(-t / tau)), 0, 40)
def run(r):
    x = S[int(r['seg'])]['vl'][int(r['w0']):int(r['w0']) + 1200]
    o = A.simulate(x, A.MPC(), preview='forecast', forecaster=fc); return {'seg': r['seg'], 'w0': r['w0'], **{f'ca_{k}': v for k, v in o.items()}}
if __name__ == '__main__':
    p, n = int(sys.argv[1]), int(sys.argv[2]); rows = [R.iloc[i].to_dict() for i in range(p, len(R), n)]
    with Pool(2) as P: out = P.map(run, rows)
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/caccel_part{p}.csv', index=False); print(len(out), 'done')
