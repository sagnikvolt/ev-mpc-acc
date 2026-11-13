"""Re-run of on-board forecaster arms under realistic plant (eta 0.65). See preregistrations/rerun_eta065_prereg.md.
Usage: python rerun_eta065.py <stage>   stage in ref|raw|tune|test|summary   (env NPROC = worker count, default 10)"""
import sys, os, time, itertools, warnings
for _k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpc_acc as A
A.set_regen(0.65)
ROOT = os.path.expanduser('~/mnt/EV Project'); OUT = f'{ROOT}/results/eta065_rerun'; os.makedirs(OUT, exist_ok=True)
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv'); E = np.load(f'{ROOT}/cache/fc_eval.npz')
GRID = list(itertools.product([0.25, 0.5, 0.75], [0.0, 0.5, 0.8])); NP = int(os.environ.get('NPROC', '5'))
TEST = [w for w in range(len(W)) if W.fold[w] > 0]; FOLD0 = [w for w in range(len(W)) if W.fold[w] == 0]
off = {w: np.where(E['win'] == w)[0] for w in np.unique(E['win'])}
_P = {}
def preds(model):
    if model not in _P:
        P = np.zeros_like(E['Y'], dtype=np.float32)
        for f in [0, 1, 2, 3, 4]:
            rows = np.load(f'{ROOT}/cache/fc_rows_f{f}.npy') if model != 'bilstm' else np.where(W.fold.values[E['win']] == f)[0]; P[rows] = np.load(f'{ROOT}/cache/fc_pred_{model}_f{f}.npy')
        _P[model] = P
    return _P[model]
def caccel_fc(h, N, dt, tau=2.0):
    v0 = h[-1]; a = np.clip((h[-1] - h[-11]) / 1.0, -3.5, 2.0) if len(h) > 11 else 0.0
    t = (np.arange(N) + 0.5) * dt; return np.clip(v0 + a * tau * (1 - np.exp(-t / tau)), 0, 40)
def run(job):
    arm, w, alpha, beta = job; r = W.iloc[w]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]; t0 = time.time()
    if arm == 'const': o = A.simulate(x, A.MPC(), preview='const')
    elif arm == 'caccel': o = A.simulate(x, A.MPC(), preview='forecast', forecaster=caccel_fc)
    elif arm == 'oracle': o = A.simulate(x, A.MPC(), preview='oracle')
    else:
        P = preds(arm.split('@')[0]); rows = off[w]; st = {'prev': None}
        def fc(h, N, dt):
            new = h[-1] + alpha * P[rows[(len(h) - 1) // 5]]
            if st['prev'] is not None and beta > 0:
                sh = np.concatenate([st['prev'][1:], st['prev'][-1:]]); new = beta * sh + (1 - beta) * new
            new = np.clip(new, 0, 40); st['prev'] = new; return new
        o = A.simulate(x, A.MPC(), preview='forecast', forecaster=fc)
    return dict(win=w, arm=arm, alpha=alpha, beta=beta, wall_s=time.time() - t0, **o)
def go(jobs, name):
    t = time.time()
    with Pool(NP, maxtasksperchild=40) as P: out = P.map(run, jobs, chunksize=1)
    pd.DataFrame(out).to_csv(f'{OUT}/{name}.csv', index=False); print(name, len(out), 'sims', round(time.time() - t), 's', flush=True)
def load(name): return pd.read_csv(f'{OUT}/{name}.csv')
if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'ref': go([('const', w, 1, 0) for w in FOLD0] + [('caccel', w, 1, 0) for w in FOLD0 + TEST], 'ref')
    elif st == 'raw': go([(m, w, 1.0, 0.0) for m in ('lgbm', 'bilstm') for w in TEST], 'raw')
    elif st == 'tune': go([(f'{m}@tune', w, a, b) for m in ('lgbm', 'bilstm') for (a, b) in GRID for w in FOLD0], 'tune')
    elif st == 'test':
        T = load('tune'); R = load('ref'); jl = R[(R.arm == 'caccel') & R.win.isin(FOLD0)].jerk_rms.mean(); sel = {}; rows = []
        for m in ('lgbm', 'bilstm'):
            g = T[T.arm == f'{m}@tune'].groupby(['alpha', 'beta']).agg(whkm=('whkm', 'mean'), short=('short_max', 'mean'), jerk=('jerk_rms', 'mean')).reset_index()
            g['score'] = g.whkm + 2 * g.short; g['ok'] = g.jerk <= jl; rows.append(g.assign(model=m)); gg = g[g.ok] if g.ok.any() else g; b = gg.loc[gg.score.idxmin()]; sel[m] = (float(b.alpha), float(b.beta))
        pd.concat(rows).to_csv(f'{OUT}/tune_grid_summary.csv', index=False); print('selected (alpha, beta) under eta 0.65:', sel, 'jerk limit', round(jl, 3), flush=True)
        jobs = []
        for m in ('lgbm', 'bilstm'):
            for tag, (a, b) in (('sel', sel[m]), ('frz', (0.5, 0.5))):
                jobs += [(f'{m}@{tag}', w, a, b) for w in TEST]
        go(jobs, 'test')
