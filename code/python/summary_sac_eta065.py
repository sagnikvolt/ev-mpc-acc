"""Summary of the SAC eta-0.65 re-run against the pre-registered rule (rerun_sac_eta065_prereg.md)."""
import os, glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.expanduser('~/mnt/EV Project'); OUT = f'{ROOT}/results/eta065_rerun'; os.chdir(ROOT)
W = pd.read_csv('results/windows.csv'); E = np.load('cache/fc_eval.npz'); terc = pd.Series(E['terc']); T = [w for w in range(len(W)) if W.fold[w] > 0]
G = 'results/gate3b'; ld = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{G}/{pat}'))]).drop_duplicates('win').set_index('win').reindex(T)
D = {'acc': pd.read_csv(f'{G}/r0_0.65_acc_0.csv').set_index('win').reindex(T), 'const': ld('r0_0.65_const_s1_n27_*.csv'), 'oracle': ld('r0_0.65_oracle_s1_n27_*.csv')}
R = pd.read_csv(f'{OUT}/ref.csv'); D['caccel'] = R[R.arm == 'caccel'].drop_duplicates('win').set_index('win').reindex(T)
best = np.minimum(D['const'].whkm_adj, D['caccel'].whkm_adj); tw = terc.reindex(T).values
for a in D: assert D[a].whkm_adj.notna().all(), f'missing windows in {a}'
S = {}
for f in sorted(glob.glob(f'{OUT}/sac_seed*_test.csv')):
    d = pd.read_csv(f); S[int(d.seed[0])] = d.set_index('win').reindex(T)
pct = lambda x, ref: 100 * (x - ref) / ref
print('SAC re-trained under eta 0.65 | 160 test windows | KE-adjusted Wh/km | changes are paired per window (median)\n')
print('%-24s %7s %9s %9s %9s %7s %6s %6s' % ('controller', 'Wh/km', 'vs ACC', 'vs MPCc', 'vs bestK', 'unsafe', 'gap m', 'jerk'))
rows = [('ACC', D['acc']), ('MPC const-speed', D['const']), ('MPC oracle', D['oracle'])] + [(f'SAC seed {s}' + (' (primary)' if s == 42 else ''), S[s]) for s in sorted(S)]
for n, x in rows:
    print('%-24s %7.2f %8.2f%% %8.2f%% %8.2f%% %7d %6.1f %6.2f' % (n, x.whkm_adj.mean(), pct(x.whkm_adj, D['acc'].whkm_adj).median(), pct(x.whkm_adj, D['const'].whkm_adj).median(), pct(x.whkm_adj, best).median(), (x.short_max > 0.5).sum(), x.gap_mean.mean(), x.jerk_rms.mean()))
print('\nPre-registered rule: SAC beats ACC only if median vs ACC <= -1.0%% AND Wilcoxon p < 0.05 AND unsafe windows <= ACC unsafe (%d).' % (D['acc'].short_max > 0.5).sum())
head = (best - D['oracle'].whkm_adj)
for s in sorted(S):
    x = S[s]; d = pct(x.whkm_adj, D['acc'].whkm_adj); p = wilcoxon(x.whkm_adj, D['acc'].whkm_adj).pvalue; un = int((x.short_max > 0.5).sum()); ua = int((D['acc'].short_max > 0.5).sum())
    ok = (d.median() <= -1.0) and (p < 0.05) and (un <= ua); sel = json.load(open(f'{OUT}/sac_seed{s}_selection.json'))
    rec = 100 * (best - x.whkm_adj).sum() / head.sum(); recd = 100 * (best - x.whkm_adj)[tw == 2].sum() / head[tw == 2].sum()
    print(f'seed {s}: ckpt {sel["selected_steps"]} steps | median vs ACC {d.median():+.2f}% mean {d.mean():+.2f}% p={p:.3g} | unsafe {un} (ACC {ua}) | headroom recovered all {rec:.1f}% dynamic {recd:.1f}% | BEATS ACC: {ok}'
          + (' | FAILED-TO-LEARN rule' if sel['failed_to_learn'] else ''))
if S:
    m = [pct(S[s].whkm_adj, D['acc'].whkm_adj).median() for s in S]; print('\nAcross seeds: median-vs-ACC min %.2f%% / median %.2f%% / max %.2f%%' % (min(m), float(np.median(m)), max(m)))
for s in sorted(S):
    print(f'\nSeed {s} fold-0 checkpoint scores:'); print(pd.read_csv(f'{ROOT}/cache/sac_eta065/seed{s}/fold0_scores.csv')[['steps', 'whkm', 'short', 'unsafe', 'score']].round(2).to_string(index=False))
