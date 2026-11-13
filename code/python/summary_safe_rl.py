"""Summary of the safe-RL experiment (safe_rl_prereg.md). Run from the project root: python python/summary_safe_rl.py > results/safe_rl/summary.txt"""
import glob, os, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
G = 'results/gate3b/'; E = 'results/eta065_rerun/'; O = 'results/safe_rl/'; W = pd.read_csv('results/windows.csv'); T = [w for w in range(len(W)) if W.fold[w] > 0]
acc = pd.read_csv(G + 'r0_0.65_acc_0.csv').set_index('win').reindex(T); const = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(G + 'r0_0.65_const_s1_n27_*.csv'))]).set_index('win').reindex(T)
nc = pd.read_csv('results/normal_car/per_window.csv').set_index('win'); gam = json.load(open(O + 'gamma.json'))['gamma']
arms = {'ACC (tuned)': acc, 'MPC const-speed': const}
for s in (42, 43, 44):
    f = f'{E}sac_seed{s}_test.csv'
    if os.path.exists(f): arms[f'plain SAC seed {s}'] = pd.read_csv(f).set_index('win').reindex(T)
for s in (42, 43, 44):
    f = f'{O}b1_seed{s}_test.csv'
    if os.path.exists(f): arms[f'B1 SAC+filter (no retrain) seed {s}'] = pd.read_csv(f).set_index('win').reindex(T)
for s in (42, 43, 44):
    f = f'{O}sac_seed{s}_test.csv'
    if os.path.exists(f): arms[f'B2 safe-SAC seed {s}' + (' (PRIMARY)' if s == 42 else '')] = pd.read_csv(f).set_index('win').reindex(T)
print('Safe RL at eta 0.65 | filter gamma %.1f | 160 test windows | KE-adjusted Wh/km | changes paired per window (median)\n' % gam)
print('%-40s %8s %8s %8s %8s %6s %7s %6s %7s %9s' % ('controller', 'Wh/km', 'vs ACC', 'vs MPCc', 'p vs ACC', 'unsafe', 'maxshort', 'gap m', 'jerk', 'filter%'))
for k, x in arms.items():
    d = 100 * (x.whkm_adj - acc.whkm_adj) / acc.whkm_adj; dm = 100 * (x.whkm_adj - const.whkm_adj) / const.whkm_adj; p = wilcoxon(x.whkm_adj, acc.whkm_adj).pvalue if k != 'ACC (tuned)' else float('nan')
    print('%-40s %8.2f %7.2f%% %7.2f%% %8.1e %6d %7.2f %6.1f %7.2f %8s' % (k, x.whkm_adj.mean(), d.median(), dm.median(), p, (x.short_max > .5).sum(), x.short_max.max(), x.gap_mean.mean(), x.jerk_rms.mean(), ('%.2f%%' % (100 * x.iv_rate.mean())) if 'iv_rate' in x else '-'))
print('\nPASS RULE (B2 seed 42 vs ACC): median <= -1.0%, Wilcoxon p<0.05, 0 unsafe windows')
k = [a for a in arms if 'PRIMARY' in a]
if k:
    x = arms[k[0]]; d = 100 * (x.whkm_adj - acc.whkm_adj) / acc.whkm_adj; p = wilcoxon(x.whkm_adj, acc.whkm_adj).pvalue; u = int((x.short_max > .5).sum())
    print('median %.2f%% (<=-1: %s) ; p %.2e (<.05: %s) ; unsafe %d (==0: %s) => %s' % (d.median(), d.median() <= -1, p, p < .05, u, u == 0, 'PASS' if (d.median() <= -1 and p < .05 and u == 0) else 'FAIL'))
    ca = nc.grp.reindex(T) == 'acc'; r = nc.rec_smooth1s.reindex(T); dd = 100 * (x.whkm_adj - r) / r
    print('vs recorded ACC-driven followers (79 windows): median %.2f%%, p %.2e' % (dd[ca].median(), wilcoxon(x.whkm_adj[ca], r[ca]).pvalue))
    print('filter infeasible steps total: %d ; mean gap safe-SAC %.1f m vs ACC %.1f m' % (x.infeasible.sum(), x.gap_mean.mean(), acc.gap_mean.mean()))
else: print('B2 not finished yet')
for s in (42, 43, 44):
    f = f'cache/safe_rl/seed{s}/fold0_scores.csv'
    if os.path.exists(f): print('\nfold-0 scores seed %d (chunk, steps, Wh/km, shortfall, unsafe, filter share, score):' % s); print(pd.read_csv(f)[['chunk', 'steps', 'whkm', 'short', 'unsafe', 'iv', 'score']].round(3).to_string(index=False))
