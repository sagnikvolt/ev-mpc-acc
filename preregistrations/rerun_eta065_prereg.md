# Pre-registration: re-run of the on-board forecaster arms under the realistic plant (eta = 0.65)

Written 2026-11-12 (Sonnet 5.5 + Roni), BEFORE any of the runs below. Motivation: Section VI-B of the paper used the earlier
full-regeneration plant (no power cap, no fade). A reviewer can reasonably say the comparison with the realistic plant is invalid.
Nothing is re-trained: forecasters keep their cached cross-fitted predictions (they do not depend on the plant).

## What is re-run (plant: regen eta 0.65, cap 35 kW, fade below 3 m/s, as in the main results)
1. Kinematic reference arms on the 160 test windows (folds 1-4): MPC constant-speed (already run at eta 0.65, reused), MPC constant-acceleration
   with 2 s decay ("caccel"), MPC oracle (already run at eta 0.65, reused). "Best kinematic" = per window the lower energy of const and caccel.
2. Raw learned forecasts in closed loop: LightGBM and BiLSTM (same windows).
3. Stabilised forecasts (R1): forecast = beta * previous forecast shifted one MPC step + (1 - beta) * (v_t + alpha * learned change).
   a) PRIMARY: alpha, beta re-tuned on fold 0 (43 windows) under eta 0.65 with the same 9-configuration grid (alpha in .25,.5,.75; beta in 0,.5,.8)
      and the same rule as before: minimise mean energy + 2 x mean max-shortfall, subject to jerk RMS not above the fold-0 constant-acceleration MPC jerk.
   b) SECONDARY: the earlier frozen setting alpha = beta = 0.5.
No weights, limits or windows are changed. SAC is handled separately (its own pre-registration).

## Pass rules (unchanged from the original, only the plant changes)
- Raw forecasts: no pass rule; report median/mean change versus best kinematic and the share of oracle headroom recovered.
- R1 (primary): pass for a model if it recovers at least 25% of the oracle headroom in the dynamic tercile AND its median change versus best kinematic is <= 0.
- Safety/comfort reported: windows with shortfall > 0.5 m, max shortfall, jerk RMS.
## Reporting
All numbers are reported whether or not they agree with the earlier run. If the conclusion changes, the paper's Section VI-B, abstract and discussion are
rewritten to match. Windows run once; results in results/eta065_rerun/.
