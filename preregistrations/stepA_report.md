# Step A — learned lead-speed forecasters (Sonnet, 2026-11-08) — for Opus review

Setup: python/fc_data.py (255,859 train samples at 1 s from all 961 segments; 42,427 eval rows at the closed-loop timing), python/fc_bench.py (ridge, KNN k=25, SVR RBF 8k-sample, LightGBM 300 trees on 8 horizons + interpolation), python/fc_lstm.py (BiLSTM 64 hidden, 7 epochs, time-capped), python/fc_score.py. Input: 10 s lead-speed history at 0.5 s. Output: lead speed over next 30 x 0.5 s. Cross-fitted: test fold f trained on all other folds (incl. tune fold and AstaZero), test days never seen. Test windows: folds 1-4 (160).

## 1. Accuracy (results/gate3b/fc_rmse.csv). RMSE m/s, overall test:
| | 1 s | 3 s | 5 s | 10 s |
|---|---|---|---|---|
| const-speed | .298 | 1.006 | 1.609 | 2.774 |
| const-accel | .158 | .707 | 1.289 | 2.519 |
| best ML (3 s / 5 s) | LGBM .148 | LGBM .684 | BiLSTM 1.257 | BiLSTM 2.445 |
Best ML vs BEST kinematic: +6.3% (1 s), +3.3% (3 s), +2.5% (5 s), +2.9% (10 s). Vs const-speed alone ~32% at 3 s, but const-accel already captures that.
**Pre-set bar (>=20% better than both kinematic at 3 s and 5 s): FAILED (3.3% / 2.5%).**
By dynamics tercile: calm: ML WORSE than const-accel (-11 to -27%); medium: ~0 or negative; dynamic: ML +4 to +7% (LGBM 7.0% at 3 s, BiLSTM 6.0% at 5 s).

## 2. Closed loop, 160 test windows (results/gate3b/ml_closedloop_summary.txt)
Energy vs best kinematic per window: oracle -1.72% median; **LGBM +2.25% (worse, better in 13/160), BiLSTM +1.58% (14/160)**. Share of oracle headroom recovered: negative everywhere, also in dynamic tercile (LGBM -103%, BiLSTM -95%). Safety: LGBM/BiLSTM 1 window shortfall (1.6/1.9 m) vs kinematic 3 windows (14.1/8.2 m); oracle 0. **Jerk RMS 0.76/0.71 vs 0.27 (const) / 0.43 (caccel) / 0.22 (oracle).**
Pipeline check: feeding const-accel through the same lookup reproduces the stored const-accel arm within 0.1% (3 windows), so no alignment bug.
Why: ML trajectories are unstable between consecutive MPC steps (mean |change| at 7.5 s / 15 s horizon: 0.54/0.60 m/s LGBM vs 0.16 const-accel); MPC chases the forecast noise (jerk, extra braking/accelerating). RMSE alone does not predict closed-loop value. Long-horizon RMSE is ~ const-accel's (3.40/3.30 vs 3.41 at 15 s).

## 3. Verdict vs Opus rules
Rule 2 (bar) FAIL. Rule 5 (<25% headroom recovered in dynamic tercile) TRIGGERED => negative/partial ML result per pre-registered fallback ("information-value + safety" study). BiLSTM and LGBM are the best models; ML is NOT best in calm traffic.
Possible rescues (not run, Opus to decide): (a) predict residual over const-accel with strong shrinkage / smoothing across MPC steps, (b) train with a closed-loop-aware or smoothness loss, (c) richer inputs, e.g. lead's own gap/speed of ITS leader (needs V2V, differs from the 'on-board only' claim), (d) lower-rate/longer-horizon features, (e) PatchTST-style model. Compute: LGBM ~25 s/fold, SVR ~30 s, KNN 4 s, BiLSTM ~150 s/fold (time cap).
