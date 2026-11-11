"""Baselines on windows. Usage: python3 python/gate3b_base.py tune <acc|smc> <cfg_start> <n>   (fold 0)  |  test <acc|smc> <cfg_idx> <slice> <size>  (folds 1-4)"""
import sys, os, itertools, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import baselines as B
ROOT = os.path.expanduser('~/mnt/EV Project'); S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv')
GRIDS = {'acc': list(itertools.product([0.05, 0.1, 0.2, 0.4], [0.3, 0.6, 1.0])), 'smc': list(itertools.product([0.1, 0.3], [0.5, 1.0, 2.0], [1.0, 2.0]))}
mode, ctrl = sys.argv[1], sys.argv[2]
def mk(c): g = GRIDS[ctrl][c]; return B.acc_policy(*g) if ctrl == 'acc' else B.smc_policy(*g)
def run(a):
    w, c = a; r = W.iloc[w]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]; return dict(win=w, cfg=c, arm=ctrl, **B.simulate(x, mk(c)))
if __name__ == '__main__':
    if mode == 'tune':
        c0, n = int(sys.argv[3]), int(sys.argv[4]); wins = [w for w in range(len(W)) if W.fold[w] == 0]; jobs = [(w, c) for c in range(c0, min(c0 + n, len(GRIDS[ctrl]))) for w in wins]; tag = f'btune_{ctrl}_{c0}'
    else:
        c, k, n = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); wins = [w for w in range(len(W)) if W.fold[w] > 0][k * n:(k + 1) * n]; jobs = [(w, c) for w in wins]; tag = f'btest_{ctrl}_{k}'
    with Pool(2) as P: out = P.map(run, jobs)
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/{tag}.csv', index=False); print(tag, len(out), 'sims')
