"""Normal car (recorded followers) vs our design car. Follows open_data_package/preregistrations/normal_car_prereg.md.
Usage (project root): python3 python/normal_car.py check   # sanity: replay of simulated MPC must match simulate()
                      python3 python/normal_car.py run     # full experiment -> results/normal_car/"""
import sys, os, glob, warnings, numpy as np, pandas as pd
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpc_acc as A
ROOT = os.path.expanduser('~/mnt/EV Project'); A.set_regen(0.65)
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv')
T = [w for w in range(len(W)) if W.fold[w] > 0]; OUT = f'{ROOT}/results/normal_car/'; N_SAMP = 209 * 5   # simulate() runs 209 steps of 0.5 s = 1045 samples; replay the same duration
def smooth(v, n):
    if n <= 1: return v.copy()
    k = np.ones(n); return np.convolve(v, k, 'same') / np.convolve(np.ones(len(v)), k, 'same')
def replay(v):
    """KE-adjusted Wh/km of a 10 Hz speed trace (same accounting as mpc_acc.simulate, dt 0.1 s)."""
    v0, v1 = v[:-1], v[1:]; vm = 0.5 * (v0 + v1)
    E = float(np.sum(A.pb_kw(A.pw_kw(v0, v1, 0.1), vm)) * 0.1 / 3.6); dist = float(np.sum(vm) * 0.1)
    dke = 0.5 * A.M * (v[-1] ** 2 - v[0] ** 2) / 3600.0
    return (E - 0.9 * dke) / dist * 1000, E / dist * 1000
def jerk_rms(v):
    a = (v[5::5] - v[:-5:5][:len(v[5::5])]) / 0.5; return float(np.sqrt(np.mean(np.square(np.diff(a) / 0.5))))
if sys.argv[1] == 'check':
    rec = {'v': []}; orig = A.pw_kw
    def pw_rec(v0, v1, dt):
        if np.ndim(v0) == 0 and dt == 0.1: rec['v'].append((float(v0), float(v1)))   # plant substeps only (solver calls use arrays / dt 0.5)
        return orig(v0, v1, dt)
    A.pw_kw = pw_rec; worst = 0
    for w in T[:3]:
        r = W.iloc[w]; x = S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200]; rec['v'].clear()
        o = A.simulate(x, A.MPC(), preview='const'); tr = np.array([rec['v'][0][0]] + [b for a, b in rec['v']])
        A.pw_kw = orig; rp = replay(tr)[0]; A.pw_kw = pw_rec
        d = 100 * abs(rp - o['whkm_adj']) / abs(o['whkm_adj']); worst = max(worst, d); print('win', w, 'simulate %.4f replay %.4f diff %.4f%%' % (o['whkm_adj'], rp, d))
    print('SANITY', 'PASS' if worst < 0.1 else 'FAIL', 'worst diff %.4f%% (limit 0.1%%)' % worst); sys.exit()
def r0(arm): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{ROOT}/results/gate3b/r0_0.65_{arm}_s1_n27_*.csv'))]).set_index('win').reindex(T)
const, orc = r0('const'), r0('oracle'); acc = pd.read_csv(f'{ROOT}/results/gate3b/r0_0.65_acc_0.csv').set_index('win').reindex(T)
mp = A.MPC(); rows = []
for w in T:
    r = W.iloc[w]; s = S[int(r.seg)]; sl = slice(int(r.w0), int(r.w0) + N_SAMP + 1)
    vf = np.clip(np.asarray(s['vf'][sl], float), 0, None); iv = np.asarray(s['ivs'][sl], float); fa = float(s['fol_acc'])
    assert len(vf) == N_SAMP + 1 and not np.isnan(vf).any()
    grp = 'acc' if fa > 0.9 else ('human' if fa < 0.1 else 'mixed')
    e1 = replay(smooth(vf, 11))[0]; eraw = replay(vf)[0]; e2 = replay(smooth(vf, 21))[0]; vs = smooth(vf, 11); ok = ~np.isnan(iv)
    gap = float(np.mean(iv[ok])) if ok.mean() > 0.5 else np.nan
    short = float(np.max(mp.dmin_a + mp.dmin_b * vs[ok] - iv[ok])) if ok.mean() > 0.5 else np.nan
    rows.append(dict(win=w, seg=int(r.seg), campaign=r.campaign, fol_veh=s['fol_veh'], grp=grp, fol_acc=fa, rec_smooth1s=e1, rec_raw=eraw, rec_smooth2s=e2,
                     rec_gap_mean=gap, rec_gap_cover=float(ok.mean()), rec_short_max=short, rec_jerk=jerk_rms(vs),
                     sim_acc=acc.whkm_adj[w], mpc_const=const.whkm_adj[w], mpc_oracle=orc.whkm_adj[w],
                     acc_gap=acc.gap_mean[w], mpc_gap=const.gap_mean[w], acc_jerk=acc.jerk_rms[w], mpc_jerk=const.jerk_rms[w], mpc_short=const.short_max[w], acc_short=acc.short_max[w], mpc_fail=const.fail[w]))
D = pd.DataFrame(rows); D.to_csv(OUT + 'per_window.csv', index=False)
def pct(a, b): return 100 * (D[a] - D[b]) / D[b]
L = []; P = L.append
P('Normal car (recorded follower, replayed through frozen BMW i3 model, eta 0.65) vs our design car. 160 test windows, KE-adjusted Wh/km. Negative = our car uses less.')
P('Groups: ACC-driven %d, human %d, mixed %d. Primary smoothing 1.0 s moving average; raw and 2.0 s are sensitivity only.\n' % tuple((D.grp == g).sum() for g in ('acc', 'human', 'mixed')))
def block(name, m):
    x = D[m]; P('%-22s n=%3d  recorded mean %.1f | MPC-const mean %.1f | median change %+.2f%% | mean change %+.2f%% | MPC lower in %d/%d | Wilcoxon p=%.2e' % (name, len(x), x.rec_smooth1s.mean(), x.mpc_const.mean(), (100 * (x.mpc_const - x.rec_smooth1s) / x.rec_smooth1s).median(), (100 * (x.mpc_const - x.rec_smooth1s) / x.rec_smooth1s).mean(), (x.mpc_const < x.rec_smooth1s).sum(), len(x), wilcoxon(x.mpc_const, x.rec_smooth1s).pvalue))
P('PRIMARY and secondary groups: MPC-const (our design car) vs recorded follower (1.0 s smoothing)')
for nm, g in (('ACC-driven (primary)', D.grp == 'acc'), ('human-driven', D.grp == 'human'), ('mixed', D.grp == 'mixed'), ('all 160', D.grp != '')): block(nm, g)
a = D.grp == 'acc'; med = (100 * (D.mpc_const - D.rec_smooth1s) / D.rec_smooth1s)[a].median(); p = wilcoxon(D.mpc_const[a], D.rec_smooth1s[a]).pvalue; uns = int((D.mpc_short[a] > 0.5).sum())
P('\nPASS RULE (79 ACC-driven windows): median <= -1.0%%: %s (%.2f%%) ; p<0.05: %s (%.2e) ; MPC unsafe windows == 0: %s (%d)  =>  %s' % (med <= -1, med, p < .05, p, uns == 0, uns, 'PASS' if (med <= -1 and p < .05 and uns == 0) else 'FAIL'))
P('\nSensitivity (ACC-driven, MPC-const vs recorded): ' + '; '.join('%s median %+.2f%%' % (k, (100 * (D.mpc_const - D[c]) / D[c])[a].median()) for k, c in (('raw', 'rec_raw'), ('1.0 s', 'rec_smooth1s'), ('2.0 s', 'rec_smooth2s'))))
P('\nOther arms vs recorded follower, median change %, by group (1.0 s smoothing):'); P('%-14s %10s %10s %10s' % ('group', 'sim ACC', 'MPC const', 'oracle'))
for nm, g in (('ACC-driven', D.grp == 'acc'), ('human', D.grp == 'human'), ('mixed', D.grp == 'mixed'), ('all', D.grp != '')):
    x = D[g]; P('%-14s %+9.2f%% %+9.2f%% %+9.2f%%' % (nm, *[(100 * (x[c] - x.rec_smooth1s) / x.rec_smooth1s).median() for c in ('sim_acc', 'mpc_const', 'mpc_oracle')]))
P('\nNOTE: group means of Wh/km are NOT comparable across groups (different roads and traffic in each group); only within-window comparisons (columns above) are valid.')
P('\nDatasheet (means over groups). gap = mean following gap (m) where recorded gap exists; jerk RMS (m/s3); unsafe = windows with shortfall > 0.5 m vs our safe-gap rule')
P('%-12s %6s %8s %8s %8s %8s %8s | %8s %8s %8s' % ('group', 'n', 'rec Wh/km', 'ACCsim', 'MPC', 'oracle', 'rec gap', 'ACC gap', 'MPC gap', 'rec jerk'))
for nm, g in (('ACC-driven', D.grp == 'acc'), ('human', D.grp == 'human'), ('mixed', D.grp == 'mixed'), ('all', D.grp != '')):
    x = D[g]; P('%-12s %6d %8.1f %8.1f %8.1f %8.1f %8.1f | %8.1f %8.1f %8.2f' % (nm, len(x), x.rec_smooth1s.mean(), x.sim_acc.mean(), x.mpc_const.mean(), x.mpc_oracle.mean(), x.rec_gap_mean.mean(), x.acc_gap.mean(), x.mpc_gap.mean(), x.rec_jerk.mean()))
P('jerk RMS: recorded %.2f | sim ACC %.2f | MPC %.2f ; unsafe windows (shortfall>0.5 m): MPC %d, sim ACC %d, recorded followers %d of %d with gap data (descriptive, different gap definition)' % (D.rec_jerk.mean(), D.acc_jerk.mean(), D.mpc_jerk.mean(), (D.mpc_short > .5).sum(), (D.acc_short > .5).sum(), (D.rec_short_max > .5).sum(), D.rec_short_max.notna().sum()))
P('recorded gap coverage: gap present (>50%% of samples) in %d of 160 windows' % D.rec_gap_mean.notna().sum())
g = D.dropna(subset=['rec_gap_mean']); P('Gap-matched check (windows with recorded gap, ACC-driven, n=%d): median MPC change %+.2f%%; recorded mean gap %.1f m vs MPC %.1f m' % ((g.grp == 'acc').sum(), (100 * (g.mpc_const - g.rec_smooth1s) / g.rec_smooth1s)[g.grp == 'acc'].median(), g.rec_gap_mean[g.grp == 'acc'].mean(), g.mpc_gap[g.grp == 'acc'].mean()))
P('\nPer follower car model (ACC-driven + human, recorded mean Wh/km, MPC change median %):')
for v, x in D.groupby('fol_veh'): P('  %-28s n=%3d recorded %.1f  MPC %+.2f%%' % (v, len(x), x.rec_smooth1s.mean(), (100 * (x.mpc_const - x.rec_smooth1s) / x.rec_smooth1s).median()))
open(OUT + 'summary.txt', 'w').write('\n'.join(L)); print('\n'.join(L))
