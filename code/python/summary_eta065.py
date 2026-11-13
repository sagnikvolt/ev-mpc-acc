"""Summary of the eta 0.65 re-run (Section VI-B). Run from the project root: python python/summary_eta065.py > results/eta065_rerun/summary.txt"""
import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
G = 'results/gate3b/'; D = 'results/eta065_rerun/'; W = pd.read_csv('results/windows.csv'); E = np.load('cache/fc_eval.npz'); terc = pd.Series(E['terc'])
T = [w for w in range(len(W)) if W.fold[w] > 0]
def r0(arm): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{G}r0_0.65_{arm}_s1_n27_*.csv'))]).set_index('win').reindex(T)
ref = pd.read_csv(D + 'ref.csv'); raw = pd.read_csv(D + 'raw.csv'); tst = pd.read_csv(D + 'test.csv')
arms = {'const': r0('const'), 'oracle': r0('oracle'), 'caccel': ref[(ref.arm == 'caccel')].set_index('win').reindex(T)}
for m in ('lgbm', 'bilstm'): arms[m + ' raw'] = raw[raw.arm == m].set_index('win').reindex(T)
for m in ('lgbm', 'bilstm'):
    for tag, lab in (('sel', 're-tuned'), ('frz', 'frozen .5/.5')):
        d = tst[tst.arm == f'{m}@{tag}']; arms[f'{m} R1 {lab} (a={d.alpha.iloc[0]}, b={d.beta.iloc[0]})'] = d.set_index('win').reindex(T)
kin = np.minimum(arms['const'].whkm_adj, arms['caccel'].whkm_adj); tw = terc.reindex(T).values
pc = lambda a: 100 * (arms[a].whkm_adj - kin) / kin; orc = pc('oracle')
print('Plant: regen eta 0.65, cap 35 kW, fade<3 m/s. TEST windows n=%d. Energy = Wh/km KE-adjusted vs best kinematic (min of const, const-accel) per window; negative = saves.' % len(T))
print('headroom recovered = mean %% change of arm / mean %% change of oracle (all windows) ; dynamic = same in the dynamic tercile\n')
print('%-42s %8s %8s %9s %9s %9s %8s %6s %6s' % ('arm', 'median%', 'mean%', 'better', 'p', 'recov.all', 'recov.dyn', 'unsafe', 'jerk'))
for a in arms:
    d = pc(a); x = arms[a]; p = wilcoxon(x.whkm_adj, kin).pvalue if a != 'oracle' else float('nan')
    rec = d.mean() / orc.mean() * 100; recd = d[tw == 2].mean() / orc[tw == 2].mean() * 100
    print('%-42s %8.2f %8.2f %5d/%3d %9.1e %8.1f%% %7.1f%% %6d %6.2f' % (a, d.median(), d.mean(), (d < 0).sum(), len(d), p, rec, recd, (x.short_max > 0.5).sum(), x.jerk_rms.mean()))
print('\nSafety: max shortfall (m): ' + ', '.join('%s %.1f' % (a, arms[a].short_max.max()) for a in arms))
for m in ('lgbm', 'bilstm'):
    for tag in ('sel', 'frz'):
        a = [k for k in arms if k.startswith(f'{m} R1') and ('re-tuned' in k) == (tag == 'sel')][0]; d = pc(a)
        print('PASS rule R1 %s %s: dynamic recovered >=25%%: %s ; median<=0: %s' % (m, tag, d[tw == 2].mean() / orc[tw == 2].mean() * 100 >= 25, d.median() <= 0))
# crude timing: wall seconds per simulated step over all sims in this re-run (includes plant simulation and Python overhead)
allr = pd.concat([ref, raw, tst]); ms = 1000 * allr.wall_s / allr.steps
print('\nWall time per MPC step (ms), %d sims, 5-7 parallel workers, includes plant simulation and Python overhead: mean %.1f, 99th pct %.1f, max %.1f (budget 500 ms)' % (len(allr), ms.mean(), ms.quantile(.99), ms.max()))
