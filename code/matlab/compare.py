import pandas as pd, numpy as np, os
R = os.path.expanduser('~/mnt/EV Project/matlab/verify'); M = pd.read_csv(f'{R}/matlab_results.csv')
A = pd.read_csv(f'{R}/data/ref_acc.csv'); P = pd.read_csv(f'{R}/data/ref_mpc.csv'); out = []
def P_(s): print(s); out.append(s)
res = {}
for arm, ref, tol_w, tol_m in [('acc', A, 0.1, None), ('mpc_const', P, 1.0, 0.5)]:
    m = M[M.arm == arm].set_index('n').sort_index(); r = ref.set_index('n').sort_index(); d = 100 * (m.whkm_adj / r.whkm_adj - 1)
    P_(f'\n{arm}: window  Python   MATLAB   diff%   dmin(py/ml)   jerk(py/ml)')
    for n in m.index: P_(f'  {n:2d}  {r.whkm_adj[n]:8.3f} {m.whkm_adj[n]:8.3f} {d[n]:+7.3f}%  {r.dmin[n]:6.2f}/{m.dmin[n]:6.2f}  {r.jerk_rms[n]:.3f}/{m.jerk_rms[n]:.3f}')
    mean_d = 100 * (m.whkm_adj.mean() / r.whkm_adj.mean() - 1); ok_w = (d.abs() <= tol_w).all(); ok_m = True if tol_m is None else abs(mean_d) <= tol_m
    P_(f'  max |diff| {d.abs().max():.3f}% (tol {tol_w}%) -> {"OK" if ok_w else "FAIL"};  mean diff {mean_d:+.3f}%' + ('' if tol_m is None else f' (tol {tol_m}%) -> {"OK" if ok_m else "FAIL"}') + f'; MATLAB solver fails: {int(m.fail.sum())}')
    res[arm] = ok_w and ok_m
m_ = M[M.arm == 'mpc_const'].set_index('n'); a_ = M[M.arm == 'acc'].set_index('n'); r_ = P.set_index('n'); ra = A.set_index('n')
P_('\nMPC vs ACC change in Wh/km (MATLAB vs Python) by window: ' + ', '.join(f'{100*(m_.whkm_adj[n]/a_.whkm_adj[n]-1):+.1f}/{100*(r_.whkm_adj[n]/ra.whkm_adj[n]-1):+.1f}' for n in m_.index))
P_('\nOVERALL: ' + ('PASS' if all(res.values()) else 'FAIL'))
open(f'{R}/compare_report.txt', 'w').write('\n'.join(out))
