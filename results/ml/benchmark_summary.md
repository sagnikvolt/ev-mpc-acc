# ML residual benchmark — BMW i3, 5-fold trip-wise CV (68 trips, 2 Hz, seed 42), Sonnet 2026-11-08

Physics model refit inside every fold; all metrics on held-out trips only. Target = P_batt(measured) − P_physics. Inputs exclude measured current/voltage/torque/SoC-derived targets.

| Model | Power RMSE kW (mean±sd over folds) | vs physics | R² | Trip-energy median abs err | p90 | Pooled err | Within 5 % | Infer µs/sample |
|---|---|---|---|---|---|---|---|---|
| Physics only (baseline) | 4.37 ± 0.48 | +0.0 % | 0.910 | 5.7 % | 15.5 % | +0.78 % | 47 % | 0.0 |
| Physics + Ridge | 4.29 ± 0.46 | -1.7 % | 0.913 | 5.1 % | 20.2 % | +0.86 % | 50 % | 0.0 |
| Physics + KNN (k=25) | 4.18 ± 0.35 | -4.4 % | 0.918 | 4.1 % | 14.8 % | -1.06 % | 56 % | 197.3 |
| Physics + SVR (RBF, 8k samples) | 4.10 ± 0.33 | -6.1 % | 0.921 | 4.8 % | 13.4 % | +0.22 % | 53 % | 231.3 |
| Physics + LightGBM | 4.03 ± 0.36 | -7.7 % | 0.923 | 4.8 % | 16.6 % | +0.91 % | 50 % | 5.0 |
| Physics + BiLSTM (10 s window) | 4.23 ± 0.72 | -3.2 % | 0.915 | 4.8 % | 17.8 % | +1.99 % | 51 % | 18.7 |
| LightGBM pure ML (no physics) | 4.14 ± 0.30 | -5.1 % | 0.919 | 4.4 % | 13.3 % | +1.04 % | 57 % | 9.7 |

## Honest reading
- **Gate 3 bar (≥30 % power-RMSE cut) is NOT met.** Best residual model (LightGBM) cuts power RMSE by ~7.7 %, SVR 6.1 %, KNN 4.3 %, BiLSTM 3.1 %. Ridge 1.8 %.
- Trip-energy median error improves more (5.7 % → 4.1–4.8 %): KNN best (4.07 %), LightGBM 4.82 %, SVR 4.77 %, BiLSTM 4.79 %. Differences between ML models are inside fold-to-fold noise (RMSE sd ≈ 0.3–0.7 kW), so **no model can be called clearly best yet**.
- **Pure LightGBM (no physics) is almost as good** (4.14 kW vs 4.03 kW residual). With random-trip CV the physics prior gives little advantage. This is the Nie & Farzaneh 2021 risk made concrete: the physics-residual claim now needs the extrapolation test to be worth anything.
- Diagnostics on residual (4.4 kW std): no timing misalignment (best lag 0 s); only 7 % is per-trip offset; ~57 % is slow (10 s-scale) and ~37 % fast; no correlation with grade/accel/speed/ambient/heating. Error is largest at high power (6.5 kW RMSE >15 kW) and in regen (5.7 kW).
- Likely limits: integer-metre elevation → noisy grade; no wind / driver / traffic-light context; regen blending not observable from our inputs. These are data limits, not model-capacity limits (BiLSTM did not beat LightGBM).

## Not yet done
- Extrapolation test (train low-speed/summer → test high-speed/winter), physics+resid vs pure ML — decides the paper's novelty claim.
- Learning curve (10k→full) and SHAP for LightGBM.
- Hyper-parameter tuning is minimal on purpose (fixed settings, one run, seed 42); no repeated seeds yet.
- Better grade signal (longer smoothing window / OpenStreetMap elevation) is a legitimate input fix, not physics tuning — worth one test.
- MATLAB re-implementation (`fitrensemble`, `fitrsvm`, `fitrnet`/LSTM) for the 'verify in MATLAB' requirement.