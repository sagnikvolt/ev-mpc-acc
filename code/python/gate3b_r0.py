"""R0 re-runs under realistic regen. Usage: python3 python/gate3b_r0.py <eta> <arm> <slice> <size>   arm in acc|smc|sac|const|oracle  (test folds 1-4, 160 windows)
Frozen configs: ACC cfg 11 (k1 .4, k2 1.0), SMC cfg 7, SAC ckpt_137034 (trained under legacy plant, evaluated only), MPC frozen weights. NO re-tuning."""
import sys, os, warnings, numpy as np, pandas as pd, torch
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); torch.set_num_threads(1)
import mpc_acc as A, baselines as B
ROOT = os.path.expanduser('~/mnt/EV Project'); S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv')
eta, arm, k, n = float(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); STRIDE = int(os.environ.get('STRIDE', '1')); A.set_regen(eta)
if arm == 'sac':
    from stable_baselines3 import SAC; import sac_env as SE; pi_sac = SE.sac_policy(SAC.load(f'{ROOT}/cache/sac/ckpt_137034.zip', device='cpu'))
def run(w):
    r = W.iloc[w]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]
    if arm == 'acc': o = B.simulate(x, B.acc_policy(0.4, 1.0))
    elif arm == 'smc': o = B.simulate(x, B.smc_policy(0.3, 0.5, 2.0))
    elif arm == 'sac': o = B.simulate(x, pi_sac)
    else: o = A.simulate(x, A.MPC(), preview=arm)
    return dict(win=w, eta=eta, arm=arm, **o)
if __name__ == '__main__':
    wins = [w for w in range(len(W)) if W.fold[w] > 0][::STRIDE][k * n:(k + 1) * n]
    with Pool(2) as P: out = P.map(run, wins)
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/r0_{eta}_{arm}_s{STRIDE}_n{n}_{k}.csv', index=False); print(eta, arm, k, len(out), 'sims')
