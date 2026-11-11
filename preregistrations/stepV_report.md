# Step V — V2V preview (Sonnet, 2026-11-09) — for Opus review. PRE-REGISTERED RULE: FAIL (as measured).

Setup (python/v2v_prep.py, v2v_fc.py, v2v_lgbm.py, v2v_run.py; results/gate3b/v2v_*.txt): ego follows vehicle k>=1 of the OpenACC platoons; V2V gives the speed of vehicle k-1 (predecessor of the lead): delay 0.2 s, 10% packet loss (hold). Windows fixed in results/v2v_windows.csv BEFORE running: test = 30 random per fold 1,2,3 (90 windows; reduced from the allowed 200 for compute: MPC sims ~3.5 s each), tune = 30 from fold 0. eta 0.65 (cap 35 kW, fade <3 m/s), all weights frozen. CACC = ACC + k3*lead accel + k4*V2V predecessor accel, grid tuned once on fold 0 (extended after the first grid hit the edge): k3 .25, k4 1.0 (interior optimum). Kinematic V2V model (lag toward extrapolated predecessor speed; Tl 6 s, tau 6 s, wa .25, fit on folds 0,-1 only). V2V-LGBM: 43 features, cross-fitted by fold; EMA alpha .5 beta .5 chosen on fold 0 (rule: lowest score, jerk <=.45; ML beat kin on fold 0).

## 1. Forecast accuracy (test, RMSE m/s; results/gate3b/v2v_rmse.txt)
| | 1 s | 3 s | 5 s | 10 s |
| best simple kinematic | .207 | .975 | 1.776 | 3.282 |
| V2V-kinematic | .289 | .845 | 1.359 | 2.730 |
| V2V-LGBM | .156 | .673 | 1.221 | 2.511 |
V2V-LGBM is 31% better than the best simple kinematic at 3 s and 5 s (bar 20%: PASS), 25% at 1 s, 24% at 10 s; positive in every test fold. V2V-kinematic: 13% / 23% (fails at 3 s). (Contrast: on-board ML beat it by only 2-3%.)

## 2. Closed loop, test 90 windows (results/gate3b/v2v_test_summary.txt)
| arm | Wh/km | median vs ACC | median vs CACC | jerk | unsafe (max short) |
| ACC | 142.16 | 0 | +0.64% | .24 | 0 |
| CACC | 141.02 | -0.63% | 0 | .41 | 19 (2.0 m) |
| MPC const-speed | 143.65 | +0.72% | +1.61% | .32 | 0 |
| MPC V2V-kinematic | 143.00 | +0.45% | +1.05% | .56 | 1 (0.5 m) |
| **MPC V2V-ML (design)** | 141.23 | **-0.34%** (p .038) | **+0.37%** (p .058) | .34 | 0 (0.3 m) |
| MPC oracle | 127.58 | -8.41% | -7.76% | .25 | 0 |
By fold (median vs ACC) V2V-ML: fold 1 +0.26%, fold 2 +0.06%, fold 3 -1.12%; CACC: +0.10, -0.28, -1.90; oracle: -4.6, -7.6, -17.4. (Folds 1-2: ~82% of lead cars are ACC-driven; fold 3 ZalaZone driver mode not recorded.)
Pre-registered PASS (>=1.0% below ACC AND >=0.5% below CACC, safety, jerk<=.45): **FAIL** (-0.34% / +0.37%). Safety and jerk OK.

## 3. Reading
1. V2V info makes forecasts genuinely better (31%), the first time ML forecasts clearly beat kinematics, yet the energy gain is only 0.3% vs ACC and none vs CACC: forecast accuracy is not the bottleneck of closed-loop value, again. Oracle headroom is big (-8.4% vs ACC; -17% on ZalaZone) so errors at 5-15 s and the way the MPC uses a noisy forecast matter.
2. CACC (simple feedforward of V2V accel) already captures -0.63% median (-1.9% on fold 3) but with 19 windows of small soft-margin shortfalls (<=2.0 m); it is the practical competitor.
3. Compute-limited sample (90 windows, 30/fold); statistics borderline (p .04/.06). More windows or a re-run with the full 200 would sharpen but unlikely to flip a -0.34% vs +0.37% result.
4. Possible levers (NOT run; Opus to decide): MPC with forecast-uncertainty awareness (e.g. use forecast only to 5 s, constant after), forecaster trained with closed-loop-aware objective, a longer-memory V2V feature set (two predecessors ahead), CACC-as-warm-start inside MPC.
