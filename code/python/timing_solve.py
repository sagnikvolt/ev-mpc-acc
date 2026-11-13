"""Clean single-process timing of every MPC QP solve call (companion measurement in rerun_sac_eta065_prereg.md).
30 evenly spaced test windows x {const, oracle} previews, eta 0.65. Run on an otherwise idle machine."""
import sys, os, time, platform, warnings
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mpc_acc as A
A.set_regen(0.65); ROOT = os.path.expanduser('~/mnt/EV Project'); OUT = f'{ROOT}/results/eta065_rerun'; os.makedirs(OUT, exist_ok=True)
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv')
TEST = [w for w in range(len(W)) if W.fold[w] > 0]; PICK = [TEST[i] for i in np.linspace(0, len(TEST) - 1, 30).astype(int)]; times = []
class TM(A.MPC):
    def solve(s, *a, **k):
        t = time.perf_counter(); r = super().solve(*a, **k); times.append((s._tag, 1000 * (time.perf_counter() - t))); return r
def fmt(s): s = int(s); return f'{s // 60:02d}:{s % 60:02d}'
if __name__ == '__main__':
    t0 = time.time(); n = 0; tot = 2 * len(PICK); print('Phase: single-core solve timing (do not use the laptop meanwhile)')
    for pv in ('const', 'oracle'):
        for w in PICK:
            r = W.iloc[w]; m = TM(); m._tag = pv; A.simulate(S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200], m, preview=pv); n += 1
            f = int(26 * n / tot); el = time.time() - t0; sys.stdout.write(f"\r[{'#' * f}{'-' * (26 - f)}] {100 * n // tot:3d}% {n}/{tot} | elapsed {fmt(el)} left ~{fmt(el * (tot - n) / n)}   "); sys.stdout.flush()
    T = pd.DataFrame(times, columns=['preview', 'ms']); T.to_csv(f'{OUT}/solve_times.csv', index=False)
    lines = [f'MPC QP solve time per call (ms), eta 0.65, single process, {len(PICK)} windows per preview; CPU: {platform.processor()} ({os.cpu_count()} logical cores), {platform.platform()}']
    for pv, g in list(T.groupby('preview')) + [('all', T)]:
        x = g.ms.values; lines.append('%-7s n=%6d mean %.1f median %.1f p95 %.1f p99 %.1f max %.1f' % (pv, len(x), x.mean(), np.median(x), np.percentile(x, 95), np.percentile(x, 99), x.max()))
    open(f'{OUT}/solve_times.txt', 'w').write('\n'.join(lines) + '\n'); print('\n' + '\n'.join(lines))
