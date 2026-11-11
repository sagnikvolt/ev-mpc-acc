"""Gate 3B step C: oracle-preview vs constant-speed-preview MPC, closed loop, 120 s windows of real lead trajectories (OpenACC pair0, human-driven lead).
Usage: python3 python/gate3b_oracle.py <part> <nparts>   -> results/gate3b/oracle_part<part>.csv"""
import sys, os, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpc_acc as A
ROOT = os.path.expanduser('~/mnt/EV Project')
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True)
W = []
for si, s in enumerate(S):
    if s['pair'] != 0 or s['lead_acc'] > 0.5 or s['campaign'] == 'ZalaZone' and False: continue
    v = s['vl']
    for w0 in range(0, len(v) - 1200 + 1, 1200):
        x = v[w0:w0 + 1200]
        if x.mean() > 5 and x.std() > 1.0: W.append((si, w0, s['campaign'], s['day'], x))
def run(a):
    si, w0, camp, day, x = a; out = dict(seg=si, w0=w0, campaign=camp, day=day, vmean=x.mean(), vstd=x.std())
    for pv in ['const', 'oracle']:
        r = A.simulate(x, A.MPC(), preview=pv); out.update({f'{pv}_{k}': v for k, v in r.items()})
    return out
if __name__ == '__main__':
    part, n = int(sys.argv[1]), int(sys.argv[2]); idx = list(range(part, len(W), n)); print(len(W), 'windows total;', len(idx), 'this part')
    os.makedirs(f'{ROOT}/results/gate3b', exist_ok=True)
    with Pool(os.cpu_count()) as p: R = p.map(run, [W[i] for i in idx])
    pd.DataFrame(R).to_csv(f'{ROOT}/results/gate3b/oracle_part{part}.csv', index=False); print('done')
