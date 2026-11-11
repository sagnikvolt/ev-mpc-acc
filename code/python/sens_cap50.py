"""Regen cap 50 kW sensitivity. Usage: python3 python/sens_cap50.py <task> where task in acc|oracle|s1b_acc|s1b_L. Does not touch earlier result files."""
import sys, os, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import mpc_acc as A, baselines as B
ROOT = os.path.expanduser('~/mnt/EV Project'); A.set_regen(0.65, cap=50.0); task = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else None; N = int(sys.argv[3]) if len(sys.argv) > 3 else None
OUT = f'{ROOT}/results/sens_cap50/{task}' + (f'_{K}' if K is not None else '') + '.csv'
if task in ('acc', 'oracle'):
    S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv')
    def run(w):
        r = W.iloc[w]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]
        o = B.simulate(x, B.acc_policy(0.4, 1.0)) if task == 'acc' else A.simulate(x, A.MPC(), preview='oracle')
        return dict(win=w, arm=task, **o)
    jobs = [w for w in range(len(W)) if W.fold[w] > 0]
else:
    import json, s1b_run as R
    A.set_regen(0.65, cap=50.0)   # s1b_run resets the cap to 35 on import
    arm = 'acc_map' if task == 's1b_acc' else 'mpc_map_L'
    jobs = [('test', arm, w) for w in R.wins('test')]; run = R.run
if __name__ == '__main__':
    if K is not None: jobs = jobs[K * N:(K + 1) * N]
    with Pool(2) as P: out = P.map(run, jobs)
    pd.DataFrame(out).to_csv(OUT, index=False); print(task, len(out))
