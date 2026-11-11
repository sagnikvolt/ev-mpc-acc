# Gate 3B step C, full rerun with Opus conditions (Sonnet, 2026-11-08) — for Opus review

Done: (1) all 203 windows; (2) day-based folds pre-registered in python/folds.py (fold 0 = TUNE, 43 windows; folds 1-4 = TEST, 160 windows; AstaZero days training-only); (3) MPC weights tuned ONCE on fold 0 with const-speed preview, rule fixed beforehand (score = Wh/km + 2*mean shortfall m, jerk<=0.45): chose w_tr=0.08, w_gap=0.2 (weights barely matter: Wh/km spread across the 9-config grid <0.3%); frozen for all arms (python/mpc_acc.py defaults). (4) unsafe windows listed; (5) forecaster step A NOT started yet (waiting on Opus because of the finding below).

## Results, TEST folds only (160 windows, results/gate3b/full_summary.txt)
| comparison | median Wh/km (KE-adj) | mean | better | p |
|---|---|---|---|---|
| const-accel vs const-speed | +0.28% | +0.75% | 41/160 | - |
| ORACLE vs const-speed | **-1.78%** | -2.76% | 150/160 | 5e-27 |
| ORACLE vs const-accel | **-2.09%** | -3.44% | 152/160 | 5e-27 |
Per test day (oracle vs best kinematic, median): Cherasco 24_9_2018 -1.47%, Cherasco/30_8_2018+Vicolungo 26_2_2019 -1.29%, ZalaZone 9_10_2019 -2.87%, Casale -1.42%.

Safety (test): const-speed and const-accel each have 3 windows with margin shortfall >0.5 m (max 14.1 m / 8.2 m), all Vicolungo 26_2_2019 (hard lead braking); no gap <3 m. Oracle: 0 windows (max 0.1 m). Jerk RMS: const .27, caccel .43, oracle .22.
Solver: 36 infeasible QP steps of 127,281 (0.03%), in 3 windows (2 Cherasco, 1 ZalaZone), identical across arms; fallback a=-1 m/s^2.

## Flag for Opus
With frozen, non-tuned-on-test weights the energy headroom is SMALLER than on the first 78 windows: oracle vs const-speed median -1.78% (<2% rule), vs const-accel -2.09% (borderline). Mean saving still ~2.8-3.4%, strongly significant, and safety benefit holds (3 unsafe windows -> 0). Realistic ML capture of 30-60% of headroom => ~0.6-1.8% median energy. Energy-only claim looks weak; safety+comfort+energy together still defensible. Decision needed: proceed to step A (forecasters), or reframe/Plan C.
