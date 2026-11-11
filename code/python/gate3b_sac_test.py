import sys, os, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); torch.set_num_threads(2)
from stable_baselines3 import SAC
import sac_env as SE, baselines as B
ROOT = SE.ROOT; m = SAC.load(f'{ROOT}/cache/sac/ckpt_137034.zip', device='cpu'); pi = SE.sac_policy(m)
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv'); R = []
for w in [w for w in range(len(W)) if W.fold[w] > 0]:
    r = W.iloc[w]; R.append(dict(win=w, arm='sac', **B.simulate(S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200], pi)))
pd.DataFrame(R).to_csv(f'{ROOT}/results/gate3b/btest_sac_0.csv', index=False); print(len(R), 'sac test sims')
