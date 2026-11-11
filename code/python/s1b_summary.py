import glob, os, pandas as pd, numpy as np
from scipy.stats import wilcoxon
R = os.path.expanduser('~/mnt/EV Project/results/gate3b'); T = {}
for f in glob.glob(f'{R}/s1b_test_*_*.csv'):
    b = os.path.basename(f)[:-4].split('_'); arm = '_'.join(b[2:-1]); T.setdefault(arm, []).append(pd.read_csv(f))
T = {k: pd.concat(v).drop_duplicates('win').set_index('win').sort_index() for k, v in T.items()}; ref = T['acc_map']; out = []
def P(s): print(s); out.append(s)
P('S1b (42 windows, mean lead speed <=20 m/s). Change vs ACC-map.')
P('%-11s %3s %8s %9s %8s %7s %7s %6s %7s %7s' % ('arm', 'n', 'Wh/km', 'vsACCmap', 'p', 'unsafe', 'shortmx', 'jerk', 'crossEx', 'dmin'))
for arm in ['acc_map', 'cacc_map', 'mpc_map', 'mpc_map_L', 'oracle_map']:
    if arm not in T: continue
    d = T[arm]; j = d.index.intersection(ref.index); pc = 100 * (d.whkm_adj[j] / ref.whkm_adj[j] - 1); p = wilcoxon(d.whkm_adj[j] - ref.whkm_adj[j]).pvalue if arm != 'acc_map' else float('nan')
    P('%-11s %3d %8.2f %+8.2f%% %8.4f %7d %7.2f %6.2f %7.2f %7.2f' % (arm, len(d), d.whkm_adj.median(), pc.median(), p, (d.short_max > 0.5).sum(), d.short_max.mean(), d.jerk_rms.mean(), d.cross_excess.mean(), d.dmin.min()))
if 'mpc_map_L' in T:
    m = T['mpc_map_L']; j = m.index.intersection(ref.index); pc = 100 * (m.whkm_adj[j] / ref.whkm_adj[j] - 1); p = wilcoxon(m.whkm_adj[j] - ref.whkm_adj[j]).pvalue
    r1 = pc.median() <= -1.0; r2 = m.cross_excess.mean() <= ref.cross_excess.mean() + 0.3; r3 = (m.short_max > 0.5).sum() <= (ref.short_max > 0.5).sum(); r4 = p < 0.05 and pc.median() < 0
    P(f'\nS1b PRIMARY RULE (MPC-map-L vs ACC-map): median dWh/km {pc.median():+.2f}% (need <=-1.0) {r1}; crossing excess {m.cross_excess.mean():.2f} vs {ref.cross_excess.mean():.2f}+0.3 {r2}; unsafe {(m.short_max>0.5).sum()} vs {(ref.short_max>0.5).sum()} {r3}; Wilcoxon p={p:.4f} {r4} => {"PASS" if all([r1,r2,r3,r4]) else "FAIL"}')
    P('windows where MPC-map-L beats ACC-map: %d/%d' % ((m.whkm_adj[j] < ref.whkm_adj[j]).sum(), len(j)))
open(f'{R}/s1b_summary.txt', 'w').write('\n'.join(out))
