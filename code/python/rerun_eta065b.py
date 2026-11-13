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
GRID = list(itertools.product([0.25, 0.5, 0.75], [0.0, 0.5, 0.8])); NP = int(os.environ.get('NPROC', '7'))
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

TITLES = {'ref': 'Reference controllers (constant-speed, constant-acceleration) at eta 0.65', 'raw': 'Raw learned forecasts: LightGBM and BiLSTM',
          'tune': 'Re-tuning stabiliser settings on the 43 tuning windows', 'test': 'Stabilised forecasts on the 160 test windows'}
ORDER = ['ref', 'raw', 'tune', 'test']; GRAND = 246 + 320 + 774 + 640; T0 = time.time(); OVERALL = {'done': 0}
def fmt(s):
    s = int(s); return f'{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}' if s >= 3600 else f'{s // 60:02d}:{s % 60:02d}'
def log(msg): open(f'{ROOT}/rerun_log3.txt', 'a').write(time.strftime('%H:%M:%S ') + msg + '\n')
def draw(name, idx, n, tot, tp):
    el = time.time() - tp; rate = n / el if n and el > 0 else 0; eta = (tot - n) / rate if rate else 0; w = 26; f = int(w * n / tot)
    sys.stdout.write(f"\rPhase {idx}/4 {name:<4}[{'#' * f}{'-' * (w - f)}] {100 * n // tot:3d}% {n}/{tot} | phase {fmt(el)} left ~{fmt(eta)} | total {fmt(time.time() - T0)} | overall {100 * (OVERALL['done'] + n) // GRAND}%  "); sys.stdout.flush()
def jobs_for(name):
    if name == 'ref': return [('const', w, 1.0, 0.0) for w in FOLD0] + [('caccel', w, 1.0, 0.0) for w in FOLD0 + TEST]
    if name == 'raw': return [(m, w, 1.0, 0.0) for m in ('lgbm', 'bilstm') for w in TEST]
    if name == 'tune': return [(f'{m}@tune', w, a, b) for m in ('lgbm', 'bilstm') for (a, b) in GRID for w in FOLD0]
    T = pd.read_csv(f'{OUT}/tune.csv'); R = pd.read_csv(f'{OUT}/ref.csv'); jl = R[(R.arm == 'caccel') & R.win.isin(FOLD0)].jerk_rms.mean(); sel = {}; rows = []
    for m in ('lgbm', 'bilstm'):
        g = T[T.arm == f'{m}@tune'].groupby(['alpha', 'beta']).agg(whkm=('whkm', 'mean'), short=('short_max', 'mean'), jerk=('jerk_rms', 'mean')).reset_index()
        g['score'] = g.whkm + 2 * g.short; g['ok'] = g.jerk <= jl; rows.append(g.assign(model=m)); gg = g[g.ok] if g.ok.any() else g; b = gg.loc[gg.score.idxmin()]; sel[m] = (float(b.alpha), float(b.beta))
    pd.concat(rows).to_csv(f'{OUT}/tune_grid_summary.csv', index=False); msg = f'selected (alpha, beta) under eta 0.65: {sel}  jerk limit {jl:.3f}'; print('\n   ' + msg); log(msg)
    return [(f'{m}@{tag}', w, a, b) for m in ('lgbm', 'bilstm') for tag, (a, b) in (('sel', sel[m]), ('frz', (0.5, 0.5))) for w in TEST]
def key(arm, w, a, b): return (str(arm), int(w), round(float(a), 6), round(float(b), 6))
def phase(name, idx):
    jobs = jobs_for(name); fin = f'{OUT}/{name}.csv'; part = f'{OUT}/{name}.partial.csv'; tp = time.time()
    if os.path.exists(fin):
        print(f'\nPhase {idx}/4 {name}: already complete, skipped'); OVERALL['done'] += len(jobs); return
    have = pd.read_csv(part) if os.path.exists(part) else pd.DataFrame(); dk = set(key(r.arm, r.win, r.alpha, r.beta) for r in have.itertuples()) if len(have) else set()
    todo = [j for j in jobs if key(*j) not in dk]; n = len(jobs) - len(todo)
    msg = f'== Phase {idx}/4: {TITLES[name]} ({len(todo)} of {len(jobs)} simulations to run)'; print('\n' + msg); log(msg); draw(name, idx, n, len(jobs), tp)
    with Pool(NP, maxtasksperchild=40) as P:
        for res in P.imap_unordered(run, todo, chunksize=1):
            pd.DataFrame([res]).to_csv(part, mode='a', header=not os.path.exists(part), index=False); n += 1; draw(name, idx, n, len(jobs), tp)
    pd.read_csv(part).to_csv(fin, index=False); os.remove(part); OVERALL['done'] += len(jobs)
    msg = f'DONE: phase {idx}/4 {name} complete in {fmt(time.time() - tp)} [OK]'; print('\n   ' + msg); log(msg)
if __name__ == '__main__':
    print(f'eta 0.65 re-run: {NP} workers, started {time.strftime("%H:%M:%S")}'); log('start')
    for i, nm in enumerate(ORDER, 1): phase(nm, i)
    msg = f'ALL 4 PHASES COMPLETE in {fmt(time.time() - T0)}'; print('\n' + msg); log(msg)
