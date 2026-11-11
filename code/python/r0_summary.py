import glob, sys, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
eta, stride = sys.argv[1], int(sys.argv[2]); W = pd.read_csv('results/windows.csv'); E = np.load('cache/fc_eval.npz'); terc = pd.Series(E['terc'])
T = [w for w in range(len(W)) if W.fold[w] > 0][::stride]; tw = terc.reindex(T).values
def load(arm):
    if arm in ('const', 'oracle'): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'results/gate3b/r0_{eta}_{arm}_s{stride}_n27_*.csv'))]).set_index('win').reindex(T)
    return pd.read_csv(f'results/gate3b/r0_{eta}_{arm}_0.csv').set_index('win').reindex(T)
D = {a: load(a) for a in ['acc', 'smc', 'sac', 'const', 'oracle']}; ref = D['const'].whkm_adj
names = {'acc': 'Conventional ACC', 'smc': 'SMC', 'sac': 'SAC (legacy-trained, eval only)', 'const': 'MPC const-speed (regen-aware)', 'oracle': 'MPC oracle preview (non-causal)'}
print(f'REGEN eta={eta}, cap 35 kW, fade<3 m/s ; windows n={len(T)} (stride {stride}) ; Wh/km KE-adj')
print('%-34s %7s %9s %8s %6s %5s %s' % ('controller', 'Wh/km', 'med vs MPC', 'gap m', 'jerk', 'unsafe', '(max shortfall m)'))
for a in D:
    d = 100 * (D[a].whkm_adj - ref) / ref; x = D[a]
    print('%-34s %7.2f %8.2f%% %8.1f %6.2f %5d (%.1f)' % (names[a], x.whkm_adj.mean(), d.median(), x.gap_mean.mean(), x.jerk_rms.mean(), (x.short_max > 0.5).sum(), x.short_max.max()))
dd = 100 * (D['oracle'].whkm_adj - D['acc'].whkm_adj) / D['acc'].whkm_adj; print('oracle vs ACC: median %.2f%% mean %.2f%% p=%.1e' % (dd.median(), dd.mean(), wilcoxon(D['oracle'].whkm_adj, D['acc'].whkm_adj).pvalue))
dm = 100 * (D['const'].whkm_adj - D['acc'].whkm_adj) / D['acc'].whkm_adj; print('MPC const vs ACC: median %.2f%% mean %.2f%% p=%.1e' % (dm.median(), dm.mean(), wilcoxon(D['const'].whkm_adj, D['acc'].whkm_adj).pvalue))
for t, l in enumerate(['calm', 'medium', 'dynamic']):
    m = tw == t; print('%-8s n=%2d median vs ACC: MPC-const %.2f%% oracle %.2f%% smc %.2f%%' % (l, m.sum(), dm[m].median(), dd[m].median(), (100 * (D['smc'].whkm_adj - D['acc'].whkm_adj) / D['acc'].whkm_adj)[m].median()))
