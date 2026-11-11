"""Tune MPC weights ONCE on fold 0 with const-speed preview. Score = mean Wh/km_adj + 2*mean(shortfall_m); require jerk_rms<=0.45.
Usage: python3 python/gate3b_tune.py <config_index_start> <n_configs> -> results/gate3b/tune_<i>.csv"""
import sys, os, itertools, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpc_acc as A
ROOT = os.path.expanduser('~/mnt/EV Project'); S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True)
W = pd.read_csv(f'{ROOT}/results/windows.csv'); W = W[W.fold == 0].reset_index(drop=True)
GRID = list(itertools.product([0.005, 0.02, 0.08], [0.02, 0.05, 0.2]))   # (w_tr, w_gap)
def run(a):
    ci, wi = a; w_tr, w_gap = GRID[ci]; r = W.iloc[wi]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]
    o = A.simulate(x, A.MPC(w_tr=w_tr, w_gap=w_gap), preview='const'); return dict(cfg=ci, w_tr=w_tr, w_gap=w_gap, win=wi, **o)
if __name__ == '__main__':
    c0, n = int(sys.argv[1]), int(sys.argv[2]); jobs = [(c, w) for c in range(c0, min(c0 + n, len(GRID))) for w in range(len(W))]
    with Pool(2) as P: out = P.map(run, jobs)
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/tune_{c0}.csv', index=False); print(len(out), 'sims')
