"""Full evaluation, frozen MPC weights (w_tr .08, w_gap .2 tuned on fold 0). Arms: const | caccel | oracle over all 203 windows.
Usage: python3 python/gate3b_full.py <slice> <slice_size> -> results/gate3b/full_<slice>.csv"""
import sys, os, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpc_acc as A
ROOT = os.path.expanduser('~/mnt/EV Project'); S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True)
W = pd.read_csv(f'{ROOT}/results/windows.csv'); ARMS = ['const', 'caccel', 'oracle']
def fc_ca(h, N, dt, tau=2.0):
    v0 = h[-1]; a = np.clip((h[-1] - h[-11]) / 1.0, -3.5, 2.0) if len(h) > 11 else 0.0
    t = (np.arange(N) + 0.5) * dt; return np.clip(v0 + a * tau * (1 - np.exp(-t / tau)), 0, 40)
def run(j):
    wi, arm = j; r = W.iloc[wi]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]
    kw = dict(preview='const') if arm == 'const' else dict(preview='oracle') if arm == 'oracle' else dict(preview='forecast', forecaster=fc_ca)
    return dict(win=wi, arm=arm, **A.simulate(x, A.MPC(), **kw))
if __name__ == '__main__':
    k, n = int(sys.argv[1]), int(sys.argv[2]); jobs = [(w, a) for w in range(len(W)) for a in ARMS][k * n:(k + 1) * n]
    with Pool(2) as P: out = P.map(run, jobs)
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/full_{k}.csv', index=False); print(k, len(out), 'sims')
