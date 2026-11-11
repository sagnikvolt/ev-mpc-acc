"""Step V runs at eta 0.65 (regen cap 35 kW, fade<3 m/s). Windows from results/v2v_windows.csv (role tune: fold 0, 30; test: folds 1-3, 90).
Usage: python3 python/v2v_run.py cacc_tune                         # ACC/CACC gain grid on tune windows (fast)
       python3 python/v2v_run.py base <test|tune>                  # ACC + chosen CACC on given role (fast)
       python3 python/v2v_run.py mpc <role> <arm> <alpha> <beta> <slice> <size>   arm in const|kin|ml|oracle   (alpha,beta used by ml only)"""
import sys, os, json, itertools, warnings, numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpc_acc as A, baselines as B, v2v_fc as V
ROOT = os.path.expanduser('~/mnt/EV Project'); A.set_regen(0.65)
S = np.load(f'{ROOT}/cache/v2v_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/v2v_windows.csv'); RX = {}
def rx_of(si):
    if si not in RX: RX[si] = V.channel(S[si]['vp'], 1000 + si)
    return RX[si]
KIN = json.load(open(f'{ROOT}/cache/v2v_kin.json')); KP = (KIN['Tl'], KIN['tau'], KIN['wa'])
E = np.load(f'{ROOT}/cache/v2v_eval.npz'); PL = np.zeros_like(E['Y'], dtype=np.float32)
for f in [0, 1, 2, 3]: PL[np.load(f'{ROOT}/cache/v2v_rows_f{f}.npy')] = np.load(f'{ROOT}/cache/v2v_pred_lgbm_f{f}.npy')
ROWS = {w: np.where(E['win'] == w)[0] for w in np.unique(E['win'])}
CG = list(itertools.product([0, 0.25, 0.5, 0.75], [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]))
def cacc_policy(k3, k4, rx, w0):
    def pi(s):
        i = w0 + s['i']; ap = np.clip((rx[i] - rx[max(i - 10, 0)]) / 1.0, -3.5, 2.0)
        return 0.4 * (s['d'] - (B.D0 + B.TH * s['v'])) + 1.0 * (s['vl'] - s['v']) + k3 * s['al'] + k4 * ap
    return pi
def win_data(wi):
    r = W.iloc[wi]; si = int(r.seg); w0 = int(r.w0); return si, w0, S[si]['vl'][w0:w0 + 1200], rx_of(si)
def run_base(a):
    wi, k3, k4 = a; si, w0, x, rx = win_data(wi); return dict(win=wi, k3=k3, k4=k4, **B.simulate(x, cacc_policy(k3, k4, rx, w0)))
def run_mpc(a):
    wi, arm, alpha, beta = a; si, w0, x, rx = win_data(wi); st = {'prev': None}; rows = ROWS[wi]
    def fc(h, N, dt):
        k = (len(h) - 1) // 5; i = w0 + len(h) - 1
        if arm == 'kin': new = h[-1] + V.kin_pred(V.feats(S[si]['vl'], rx, i)[None, :].astype(np.float64), KP)[0]
        else:
            new = h[-1] + alpha * PL[rows[k]]
            if st['prev'] is not None and beta > 0: new = beta * np.concatenate([st['prev'][1:], st['prev'][-1:]]) + (1 - beta) * new
        new = np.clip(new, 0, 40); st['prev'] = new; return new
    kw = dict(preview='const') if arm == 'const' else dict(preview='oracle') if arm == 'oracle' else dict(preview='forecast', forecaster=fc)
    return dict(win=wi, arm=arm, alpha=alpha, beta=beta, **A.simulate(x, A.MPC(), **kw))
if __name__ == '__main__':
    mode = sys.argv[1]
    with Pool(2) as P:
        if mode == 'cacc_tune':
            wins = list(W.index[W.role == 'tune']); out = P.map(run_base, [(w, k3, k4) for w in wins for k3, k4 in CG]); T = pd.DataFrame(out)
            g = T.groupby(['k3', 'k4']).agg(whkm=('whkm_adj', 'mean'), short=('short_max', 'mean'), jerk=('jerk_rms', 'mean')).reset_index(); g['score'] = g.whkm + 2 * g.short; g['ok'] = g.jerk <= 0.487
            print(g.round(3).to_string()); b = g[g.ok].sort_values('score').iloc[0]; json.dump(dict(k3=float(b.k3), k4=float(b.k4)), open(f'{ROOT}/cache/v2v_cacc.json', 'w')); print('CHOSEN CACC', b.k3, b.k4)
        elif mode == 'base':
            role = sys.argv[2]; wins = list(W.index[W.role == role]); c = json.load(open(f'{ROOT}/cache/v2v_cacc.json'))
            out = P.map(run_base, [(w, 0.0, 0.0) for w in wins] + [(w, c['k3'], c['k4']) for w in wins]); T = pd.DataFrame(out); T['arm'] = np.where((T.k3 == 0) & (T.k4 == 0), 'acc', 'cacc'); T.to_csv(f'{ROOT}/results/gate3b/v2v_base_{role}.csv', index=False)
        else:
            role, arm, alpha, beta, k, n = sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
            wins = list(W.index[W.role == role])[k * n:(k + 1) * n]; out = P.map(run_mpc, [(w, arm, alpha, beta) for w in wins])
            pd.DataFrame(out).to_csv(f'{ROOT}/results/gate3b/v2v_{role}_{arm}_a{alpha}_b{beta}_{k}.csv', index=False); print(role, arm, alpha, beta, k, len(out), 'sims')
