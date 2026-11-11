"""Indian-scenario runs per admin/indian_prereg.md. Usage: python3 python/scen_run.py <scenario 1|2|3> <arm> <slice> <size>
arms: acc cacc smc acc_map mpc_const mpc_map mpc_v2v_ml oracle. Windows = even-position rows of role=test. Seed = 7000+100*scenario+window row."""
import sys, os, json, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
import mpc_acc as A, baselines as B, v2v_fc as V, scenarios as SC
ROOT = os.path.expanduser('~/mnt/EV Project'); A.set_regen(0.65)
S = np.load(f'{ROOT}/cache/v2v_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/v2v_windows.csv'); CC = json.load(open(f'{ROOT}/cache/v2v_cacc.json'))
TEST = list(W.index[W.role == 'test'])[0::2]; MODELS = {}
def models(f):
    if f not in MODELS: MODELS[f] = joblib.load(f'{ROOT}/cache/v2v_models_f{f}.pkl'); [m.set_params(n_jobs=1) for m in MODELS[f]['M']]
    return MODELS[f]
def run(a):
    sc, arm, wi = a; r = W.loc[wi]; si = int(r.seg); w0 = int(r.w0); x0 = S[si]['vl'][w0:w0 + 1200]; rx = V.channel(S[si]['vp'], 1000 + si)[w0:w0 + 1200]
    vl, ev, bk = SC.GEN[sc](x0, np.random.default_rng(7000 + 100 * sc + wi)); kw = dict(gap_events=ev, breakers=bk)
    if arm in ('acc', 'cacc', 'smc', 'acc_map'):
        if arm == 'acc': pol = B.acc_policy(.4, 1.0)
        elif arm == 'smc': pol = B.smc_policy(.3, .5, 2.0)
        elif arm == 'acc_map': pol = B.acc_map_policy(.4, 1.0)
        else:
            def pol(s):
                i = s['i']; ap = np.clip((rx[i] - rx[max(i - 10, 0)]) / 1.0, -3.5, 2.0)
                return 0.4 * (s['d'] - (B.D0 + B.TH * s['v'])) + 1.0 * (s['vl'] - s['v']) + CC['k3'] * s['al'] + CC['k4'] * ap
        o = B.simulate(vl, pol, **kw)
    elif arm == 'oracle': o = A.simulate(vl, A.MPC(), preview='oracle', **kw)
    elif arm in ('mpc_const', 'mpc_map'): o = A.simulate(vl, A.MPC(), preview='const', use_map=(arm == 'mpc_map'), **kw)
    else:
        M = models(int(r.fold)); st = {'p': None}
        def fc(h, N, dt):
            i = len(h) - 1; x = V.feats(vl, rx, i)[None, :].astype(np.float64); p = np.interp(np.arange(30), M['J'], [m.predict(x)[0] for m in M['M']])
            new = h[-1] + 0.5 * p
            if st['p'] is not None: new = 0.5 * np.concatenate([st['p'][1:], st['p'][-1:]]) + 0.5 * new
            new = np.clip(new, 0, 40); st['p'] = new; return new
        o = A.simulate(vl, A.MPC(), preview='forecast', forecaster=fc, **kw)
    return dict(sc=sc, arm=arm, win=wi, **o)
if __name__ == '__main__':
    sc, arm, k, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); wins = TEST[k * n:(k + 1) * n]
    with Pool(2) as P: out = P.map(run, [(sc, arm, w) for w in wins])
    pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/scen_s{sc}_{arm}_{k}.csv', index=False); print(sc, arm, k, len(out))
