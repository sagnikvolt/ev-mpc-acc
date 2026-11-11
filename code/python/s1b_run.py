"""S1b per admin/indian_prereg_s1b.md. Usage: python3 python/s1b_run.py <role test|tune> <arm> <slice> <size> ; arms acc_map cacc_map mpc_map mpc_map_L oracle_map
Test windows: role=test with mean lead speed <=20 m/s (42). Seeds 7100+row (same as S1)."""
import sys, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import mpc_acc as A, baselines as B, v2v_fc as V, scenarios as SC
ROOT = os.path.expanduser('~/mnt/EV Project'); A.set_regen(0.65)
S = np.load(f'{ROOT}/cache/v2v_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/v2v_windows.csv'); CC = json.load(open(f'{ROOT}/cache/v2v_cacc.json'))
def wins(role):
    idx = list(W.index[W.role == role]); return [w for w in idx if S[int(W.loc[w].seg)]['vl'][int(W.loc[w].w0):int(W.loc[w].w0) + 1200].mean() <= 20.0]
def run(a):
    role, arm, wi = a; r = W.loc[wi]; si = int(r.seg); w0 = int(r.w0); x0 = S[si]['vl'][w0:w0 + 1200]
    vl, _, bk = SC.s1_breakers(x0, np.random.default_rng(7100 + wi)); vp = S[si]['vp'][w0:w0 + 1200]
    xl0 = 5.0 + 1.5 * x0[0]; vpe = SC.apply_caps(vp, xl0 + 5.0 + 1.5 * vp[0], bk); rx = V.channel(vpe, 1000 + si)
    kw = dict(breakers=bk)
    if arm == 'acc_map': o = B.simulate(vl, B.acc_map_policy(.4, 1.0), **kw)
    elif arm == 'cacc_map':
        def base(s):
            i = s['i']; ap = np.clip((rx[i] - rx[max(i - 10, 0)]) / 1.0, -3.5, 2.0)
            return 0.4 * (s['d'] - (B.D0 + B.TH * s['v'])) + 1.0 * (s['vl'] - s['v']) + CC['k3'] * s['al'] + CC['k4'] * ap
        cap = B.acc_map_policy(.4, 1.0)
        def pol(s):
            a = base(s); xs = [x for x in s['breakers'] if x > s['xe']]
            if xs: a = min(a, 1.0 * (np.sqrt(s['vb'] ** 2 + 2 * 1.5 * max(min(xs) - s['xe'], 0.0)) - s['v']))
            return a
        o = B.simulate(vl, pol, **kw)
    elif arm == 'mpc_map': o = A.simulate(vl, A.MPC(), preview='const', use_map=True, **kw)
    elif arm == 'mpc_map_L': o = A.simulate(vl, A.MPC(), preview='constmap', use_map=True, **kw)
    elif arm == 'oracle_map': o = A.simulate(vl, A.MPC(), preview='oracle', use_map=True, **kw)
    return dict(role=role, arm=arm, win=wi, **o)
if __name__ == '__main__':
    role, arm, k, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); ws = wins(role)[k * n:(k + 1) * n]
    with Pool(2) as P: out = P.map(run, [(role, arm, w) for w in ws])
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/s1b_{role}_{arm}_{k}.csv', index=False); print(role, arm, k, len(out))
