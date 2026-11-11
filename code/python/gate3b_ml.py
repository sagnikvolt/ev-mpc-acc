"""Closed-loop with learned forecasts (precomputed cross-fitted predictions, looked up by (window, step); steps align exactly with eval set).
Usage: python3 python/gate3b_ml.py <model> <slice> <size> -> results/gate3b/ml_<model>_<slice>.csv  (test folds 1-4 only, 160 windows)"""
import sys, os, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpc_acc as A
ROOT = os.path.expanduser('~/mnt/EV Project'); S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True)
W = pd.read_csv(f'{ROOT}/results/windows.csv'); E = np.load(f'{ROOT}/cache/fc_eval.npz'); model = sys.argv[1]
P = np.zeros_like(E['Y'], dtype=np.float32)
for f in [1, 2, 3, 4]:
    rows = np.load(f'{ROOT}/cache/fc_rows_f{f}.npy') if model != 'bilstm' else np.where(W.fold.values[E['win']] == f)[0]; P[rows] = np.load(f'{ROOT}/cache/fc_pred_{model}_f{f}.npy')
off = {w: np.where(E['win'] == w)[0] for w in np.unique(E['win'])}
def run(w):
    r = W.iloc[w]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]; rows = off[w]
    fc = lambda h, N, dt: np.clip(h[-1] + P[rows[(len(h) - 1) // 5]], 0, 40)
    return dict(win=w, arm=model, **A.simulate(x, A.MPC(), preview='forecast', forecaster=fc))
if __name__ == '__main__':
    k, n = int(sys.argv[2]), int(sys.argv[3]); wins = [w for w in range(len(W)) if W.fold[w] > 0][k * n:(k + 1) * n]
    with Pool(2) as Pl: out = Pl.map(run, wins)
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/ml_{model}_{k}.csv', index=False); print(model, k, len(out), 'sims')
