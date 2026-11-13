"""Step D: datasheet 'normal car' vs 'our design car' (run from project root; reads existing result files only) -> results/datasheet_normal_vs_design.md/.csv"""
import glob, os, numpy as np, pandas as pd
G = 'results/gate3b/'; W = pd.read_csv('results/windows.csv'); T = [w for w in range(len(W)) if W.fold[w] > 0]
nc = pd.read_csv('results/normal_car/per_window.csv').set_index('win').reindex(T)
r = lambda f: pd.read_csv(f).set_index('win').reindex(T)
arms = {'Tuned ACC (simulated, conventional controller)': r(G + 'r0_0.65_acc_0.csv'), 'Sliding-mode controller (SMC)': r(G + 'r0_0.65_smc_0.csv'),
        'Plain SAC (RL, seed 42)': r('results/eta065_rerun/sac_seed42_test.csv'), 'Our design car: battery-aware MPC (causal)': pd.concat([pd.read_csv(f) for f in sorted(glob.glob(G + 'r0_0.65_const_s1_n27_*.csv'))]).set_index('win').reindex(T),
        'MPC oracle (non-causal ceiling)': pd.concat([pd.read_csv(f) for f in sorted(glob.glob(G + 'r0_0.65_oracle_s1_n27_*.csv'))]).set_index('win').reindex(T)}
if os.path.exists('results/safe_rl/sac_seed42_test.csv'): arms['Safe SAC (RL + safety filter, seed 42)'] = r('results/safe_rl/sac_seed42_test.csv')
st = pd.read_csv('results/eta065_rerun/solve_times.csv').ms; ct = {'MPC': 'median %.0f ms, p99 %.0f ms' % (st.median(), st.quantile(.99))}
comp = lambda k: 'n/a (needs the lead car\'s future)' if 'oracle' in k else (ct['MPC'] if 'MPC' in k else 'not timed (simple formula / small network)')
rows = []
def panel(name, mask):
    rec = nc.rec_smooth1s[mask]; base = rec.mean()
    rows.append(dict(panel=name, vehicle='Normal car: recorded real follower (replayed through the same battery model)', n=int(mask.sum()), whkm=rec.mean(), vs_recorded='0 (reference)', gap_m=nc.rec_gap_mean[mask].mean(), jerk=nc.rec_jerk[mask].mean(), unsafe='n/a', compute='-'))
    for k, x in arms.items():
        x = x[mask.values]; d = 100 * (x.whkm_adj.values - rec.values) / rec.values
        rows.append(dict(panel=name, vehicle=k, n=int(mask.sum()), whkm=x.whkm_adj.mean(), vs_recorded='%+.2f%% (median)' % np.median(d), gap_m=x.gap_mean.mean(), jerk=x.jerk_rms.mean(), unsafe=int((x.short_max > .5).sum()), compute=comp(k)))
panel('All 160 test windows', nc.grp != '')
panel('79 windows with a commercial ACC follower', nc.grp == 'acc')
D = pd.DataFrame(rows); D.to_csv('results/datasheet_normal_vs_design.csv', index=False)
L = ['# Datasheet: normal car vs our design car (battery model: BMW i3 physics, regen efficiency 0.65, 35 kW cap; KE-adjusted Wh/km; day-held-out test windows)\n',
     'Normal car = the follower vehicle actually recorded in OpenACC (commercial ACC or human), its speed trace replayed through the same battery model, 1 s smoothing (pre-registered). Gap = mean following gap in metres (recorded gap missing in 40 windows, mostly human-driven). Jerk = RMS, m/s^3. Unsafe = windows with gap shortfall > 0.5 m against the safe-gap rule 4 + 0.7 v (not computed for recorded cars: different gap definition). Wh/km means are over the windows in the panel; the % column is the paired per-window median change versus the recorded follower (negative = uses less).\n']
for p, x in D.groupby('panel', sort=False):
    L += ['## ' + p + ' (n = %d)\n' % x.n.iloc[0], '| Vehicle / controller | Mean Wh/km | Change vs recorded car | Mean gap (m) | Jerk RMS | Unsafe windows | Compute per decision |', '|---|---|---|---|---|---|---|']
    for _, r in x.iterrows(): L.append('| %s | %.1f | %s | %.1f | %.2f | %s | %s |' % (r.vehicle, r.whkm, r.vs_recorded, r.gap_m, r.jerk, r.unsafe, r.compute))
    L.append('')
L += ['Notes: (1) MPC compute time measured on an i5-12500H, one core (30 windows, 12,540 solves); the control step is 500 ms. ACC, SMC and SAC decision times were not timed; they are a simple formula or a small network. (2) The oracle knows the lead car\'s future and is a ceiling, not a deployable car. (3) The tuned ACC saves slightly more than the causal MPC against the recorded cars; the MPC\'s advantage appears only with preview (oracle). (4) Gap matters: in the commercial-ACC panel the tuned ACC and SAC follow at about 50-59 m versus 40 m for the recorded cars and 41 m for our MPC; a longer gap lets a car drive more smoothly, so part of their saving is the larger gap. The MPC compares with the recorded cars at the same gap. (5) Mean Wh/km differs between panels because the windows differ.']
open('results/datasheet_normal_vs_design.md', 'w').write('\n'.join(L)); print('\n'.join(L))
