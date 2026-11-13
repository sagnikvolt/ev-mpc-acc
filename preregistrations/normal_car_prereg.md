# Pre-registration: "normal car" (real recorded followers) vs our design car

Written 2026-11-13 (Opus 5.5 + Roni), BEFORE any energy of the recorded follower traces was computed. Only descriptive counts were looked at
(160 test windows: follower ACC-driven 79, human-driven 40, mixed 41; 8 follower car models; follower speed complete in all windows; gap missing in part of 40 windows).

## Question
Does our design car use less battery energy than real cars driving behind the same real lead vehicles?
"Normal car" = the follower vehicle actually recorded in OpenACC (commercial ACC engaged, or a human driver).
"Our design car" (primary) = the causal battery-aware MPC with constant-speed preview (no map, no V2V), already run at eta 0.65 (results/gate3b/r0_0.65_const_*).
Motivation: the OpenACC authors report commercial ACC followers using 2.7-20.5% more tractive energy than human drivers.

## Method (fixed)
- Same vehicle for every arm: each recorded follower speed trace (vf, 10 Hz, 120 s window) is replayed through the frozen BMW i3 energy model at eta 0.65
  (cap 35 kW, fade 3 m/s; mpc_acc.pb_kw / pw_kw, 0.1 s steps). This compares DRIVING BEHAVIOUR, not the different cars' powertrains.
- Energy metric identical to all other experiments: Wh/km at the battery, kinetic-energy adjusted (90% credit), with the trace's own start speed vf[0].
- Speed-signal noise would inflate the replayed energy of recorded traces (spurious accelerations). PRIMARY: recorded speed smoothed with a centred 1.0 s moving average
  (11 samples, edges shortened) before replay. SENSITIVITY (reported, no decision): raw signal and 2.0 s moving average. Our simulated arms are not smoothed (they have no sensor noise).
- Windows: the 160 day-held-out test windows (folds 1-4). No window is dropped.
- Arms reported: recorded follower (all / ACC-driven fol_acc>0.9 / human fol_acc<0.1 / mixed), our tuned simulated ACC (r0_0.65_acc), MPC constant-speed (our design car),
  MPC oracle (ceiling, non-causal). Secondary on the 90 V2V test windows: MPC with V2V LightGBM forecast (existing results) vs recorded follower.
- Also reported: mean gap and gap shortfall of the recorded followers where IVS exists (descriptive; gap definitions differ from our controller's), jerk RMS of the smoothed traces.

## Pass rule (primary, one test)
Our design car (MPC const) BEATS the real commercial-ACC car if, on the 79 windows with an ACC-driven follower:
median per-window change in KE-adjusted Wh/km <= -1.0%, AND Wilcoxon signed-rank p < 0.05, AND MPC has 0 unsafe windows (shortfall > 0.5 m) in those windows.
Secondary comparisons (human drivers, all 160, oracle, simulated ACC vs recorded ACC) are reported with medians and p-values but carry no pass rule.

## Honesty notes (fixed in advance)
- The recorded cars chose their own gap; a car keeping a longer gap can drive more smoothly. Mean gaps are reported next to the energy result.
- If the result fails, it is reported as failed. The smoothing window is not changed after seeing results.
- Output: results/normal_car/ (per-window CSV, summary.txt). Script: python/normal_car.py.
