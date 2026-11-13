"""Step B1 + gamma pick (safe_rl_prereg.md). gamma chosen on fold 0 with SAC seed 42 (eta 0.65, selected ckpt) + filter at evaluation; then the 3 existing seeds + filter on test (B1).
Usage: python3 python/safe_rl_gamma.py pick | python3 python/safe_rl_gamma.py test <seed>"""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(1)
from stable_baselines3 import SAC
import mpc_acc as A, baselines as B, safe_filter as SF
ROOT = os.path.expanduser('~/mnt/EV Project'); OUT = f'{ROOT}/results/safe_rl'; A.set_regen(0.65)
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv')
F0 = [w for w in range(len(W)) if W.fold[w] == 0]; TEST = [w for w in range(len(W)) if W.fold[w] > 0]
def load(seed):
    sel = json.load(open(f'{ROOT}/results/eta065_rerun/sac_seed{seed}_selection.json')); return SAC.load(f'{ROOT}/cache/sac_eta065/seed{seed}/ckpt_{sel["selected_steps"]}.zip', device='cpu')
def evaluate(m, wins, gamma):
    R = []
    for w in wins:
        r = W.iloc[w]; pi = SF.filtered_policy(m, gamma); o = B.simulate(S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200], pi)
        R.append(dict(win=w, **o, iv_rate=pi.iv / pi.n, infeasible=pi.inf))
    return pd.DataFrame(R)
if sys.argv[1] == 'pick':
    m = load(42); rows = []
    for g in (0.1, 0.3, 0.5):
        R = evaluate(m, F0, g); rows.append(dict(gamma=g, whkm=R.whkm_adj.mean(), unsafe=int((R.short_max > 0.5).sum()), short_max=R.short_max.max(), iv_rate=R.iv_rate.mean(), infeasible=int(R.infeasible.sum()), gap=R.gap_mean.mean())); print(rows[-1], flush=True)
    P = pd.DataFrame(rows); P.to_csv(f'{OUT}/gamma_pick.csv', index=False); ok = P[P.unsafe == 0]
    g = float(ok.loc[ok.whkm.idxmin(), 'gamma']) if len(ok) else float(P.gamma.min()); json.dump(dict(gamma=g, rule='lowest fold-0 Wh/km with 0 unsafe, else smallest gamma'), open(f'{OUT}/gamma.json', 'w')); print('CHOSEN gamma', g)
else:
    seed = int(sys.argv[2]); g = json.load(open(f'{OUT}/gamma.json'))['gamma']; R = evaluate(load(seed), TEST, g); R.insert(1, 'seed', seed); R.insert(2, 'gamma', g); R.to_csv(f'{OUT}/b1_seed{seed}_test.csv', index=False)
    print('seed', seed, 'gamma', g, 'whkm', R.whkm_adj.mean().round(2), 'unsafe', int((R.short_max > .5).sum()), 'iv_rate', R.iv_rate.mean().round(3), 'infeasible', int(R.infeasible.sum()))
