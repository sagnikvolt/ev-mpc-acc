import glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = os.path.expanduser('~/mnt/EV Project/results/gate3b'); D = {}
for f in glob.glob(f'{R}/scen_s*_*_*.csv'):
    b = os.path.basename(f)[:-4].split('_'); sc = int(b[1][1:]); arm = '_'.join(b[2:-1]); D.setdefault((sc, arm), []).append(pd.read_csv(f))
T = {k: pd.concat(v).drop_duplicates('win').set_index('win').sort_index() for k, v in D.items()}
out = []
def P(s): print(s); out.append(s)
for sc in (1, 2, 3):
    P(f'\n=== S{sc} ==='); ref = T[(sc, 'acc')]
    P('%-11s %3s %8s %9s %8s %7s %7s %6s' % ('arm', 'n', 'Wh/km', 'vsACC%', 'p', 'unsafe', 'shortmx', 'jerk') + ('  crossEx  crossMax' if sc == 1 else ''))
    for arm in ['acc', 'cacc', 'smc', 'acc_map', 'mpc_const', 'mpc_map', 'mpc_v2v_ml', 'oracle']:
        if (sc, arm) not in T: continue
        d = T[(sc, arm)]; j = d.index.intersection(ref.index); pc = 100 * (d.whkm_adj[j] / ref.whkm_adj[j] - 1); p = wilcoxon(d.whkm_adj[j] - ref.whkm_adj[j]).pvalue if arm != 'acc' else float('nan')
        P('%-11s %3d %8.2f %+8.2f%% %8.3f %7d %7.2f %6.2f' % (arm, len(d), d.whkm_adj.median(), pc.median(), p, (d.short_max > 0.5).sum(), d.short_max.mean(), d.jerk_rms.mean()) + ('  %6.2f  %6.2f' % (d.cross_excess.mean(), d.cross_excess_max.mean()) if sc == 1 else ''))
a, m = T[(1, 'acc_map')], T[(1, 'mpc_map')]; j = a.index.intersection(m.index); pc = 100 * (m.whkm_adj[j] / a.whkm_adj[j] - 1); p = wilcoxon(m.whkm_adj[j] - a.whkm_adj[j]).pvalue
r1 = pc.median() <= -1.0; r2 = m.cross_excess.mean() <= a.cross_excess.mean() + 0.3; r3 = (m.short_max > 0.5).sum() <= (a.short_max > 0.5).sum(); r4 = p < 0.05 and pc.median() < 0
P(f'\nPRIMARY RULE (MPC-map vs ACC-map): median dWh/km {pc.median():+.2f}% (need <= -1.0) {r1}; crossing excess {m.cross_excess.mean():.2f} vs {a.cross_excess.mean():.2f}+0.3 {r2}; unsafe {(m.short_max>0.5).sum()} vs {(a.short_max>0.5).sum()} {r3}; Wilcoxon p={p:.4f} {r4}  => {"PASS" if all([r1,r2,r3,r4]) else "FAIL"}')
open(f'{R}/scenarios_summary.txt', 'w').write('\n'.join(out))
