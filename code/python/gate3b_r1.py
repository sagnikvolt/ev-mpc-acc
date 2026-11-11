"""Rescue R1: forecast = beta*(previous forecast shifted one MPC step) + (1-beta)*(v_t + alpha*ML_rel).
Usage: python3 python/gate3b_r1.py tune <model> <cfg_start> <n_cfg>   (fold-0 windows, 9-config grid)
       python3 python/gate3b_r1.py test <model> <alpha> <beta> <slice> <size>  (test folds 1-4)"""
import sys, os, itertools, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpc_acc as A
ROOT = os.path.expanduser('~/mnt/EV Project'); S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True)
W = pd.read_csv(f'{ROOT}/results/windows.csv'); E = np.load(f'{ROOT}/cache/fc_eval.npz')
mode, model = sys.argv[1], sys.argv[2]; GRID = list(itertools.product([0.25, 0.5, 0.75], [0.0, 0.5, 0.8]))
P = np.zeros_like(E['Y'], dtype=np.float32)
for f in [0, 1, 2, 3, 4]:
    rows = np.load(f'{ROOT}/cache/fc_rows_f{f}.npy') if model != 'bilstm' else np.where(W.fold.values[E['win']] == f)[0]; P[rows] = np.load(f'{ROOT}/cache/fc_pred_{model}_f{f}.npy')
off = {w: np.where(E['win'] == w)[0] for w in np.unique(E['win'])}
def run(a):
    w, alpha, beta = a; r = W.iloc[w]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]; rows = off[w]; st = {'prev': None}
    def fc(h, N, dt):
        new = h[-1] + alpha * P[rows[(len(h) - 1) // 5]]
        if st['prev'] is not None and beta > 0:
            sh = np.concatenate([st['prev'][1:], st['prev'][-1:]]); new = beta * sh + (1 - beta) * new
        new = np.clip(new, 0, 40); st['prev'] = new; return new
    return dict(win=w, alpha=alpha, beta=beta, arm=model, **A.simulate(x, A.MPC(), preview='forecast', forecaster=fc))
if __name__ == '__main__':
    if mode == 'tune':
        c0, n = int(sys.argv[3]), int(sys.argv[4]); wins = [w for w in range(len(W)) if W.fold[w] == 0]
        jobs = [(w, *GRID[c]) for c in range(c0, min(c0 + n, 9)) for w in wins]; tag = f'r1tune_{model}_{c0}'
    else:
        alpha, beta, k, n = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
        wins = [w for w in range(len(W)) if W.fold[w] > 0][k * n:(k + 1) * n]; jobs = [(w, alpha, beta) for w in wins]; tag = f'r1test_{model}_{k}'
    with Pool(2) as Pl: out = Pl.map(run, jobs)
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/{tag}.csv', index=False); print(tag, len(out), 'sims')
