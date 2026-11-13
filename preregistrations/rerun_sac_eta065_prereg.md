# Pre-registration: SAC (reinforcement-learning) baseline re-trained under the realistic plant (eta = 0.65)

Written 2026-11-12 (Sonnet 5.5 + Roni), BEFORE any training or evaluation below. Motivation: the SAC baseline in the paper was trained under the earlier
full-regeneration plant (free braking energy) and only evaluated at eta 0.65 (r0_report.md, caveat 4). A reviewer can say the RL baseline was trained on the wrong physics.
Nothing else changes: same environment, reward, network, hyper-parameters, training folds, windows and metrics as python/sac_env.py / sac_train.py.

## What changes
- `mpc_acc.set_regen(0.65)` (cap 35 kW, fade below 3 m/s) is active during training, validation and test, i.e. the energy term of the reward now sees the realistic plant.
- Training data: segments of folds 0 and -1 only (test folds 1-4 never seen), as before.

## Procedure (fixed in advance)
1. Three independent training runs, seeds 42 (PRIMARY, same seed as the original), 43 and 44 (robustness). SAC: MlpPolicy [128,128], lr 3e-4, buffer 200k, batch 256,
   learning_starts 2000, gamma 0.98, tau 0.01 (unchanged).
2. Each run trains for 200,000 environment steps. A checkpoint is saved every 20,000 steps (10 checkpoints per seed).
3. Each checkpoint is scored on fold 0 (43 windows, deterministic policy): score = mean KE-adjusted Wh/km + 2 x mean max-shortfall (same rule as the original).
   The checkpoint with the lowest fold-0 score is selected, per seed. Test windows are never used for selection.
4. The selected checkpoint of each seed is run ONCE on the 160 test windows (folds 1-4). No retuning afterwards, no extra seeds, no extra steps.
5. Collision-collapse rule (the original v1 run learned to crash early): if every checkpoint of a seed has a fold-0 mean shortfall above 5 m, that seed is reported as
   "failed to learn safe following" and its test results are still reported, not hidden. The reward is not edited after seeing results.

## Question and rule (fixed before running)
Claim being tested (paper): at this training budget, RL adds nothing over a plain ACC controller and is less safe.
- SAC "beats ACC" only if ALL of: median change in KE-adjusted Wh/km vs conventional ACC (r0_0.65_acc) is <= -1.0%; Wilcoxon signed-rank p < 0.05 (160 paired windows);
  number of unsafe windows (max shortfall > 0.5 m) is not larger than ACC's.
- Primary decision uses seed 42. Seeds 43 and 44 are reported next to it (median across seeds, and min/max). If seed 42 and the other seeds disagree on the verdict, the paper says so.
- Also reported (no pass rule): change vs MPC constant-speed, vs best kinematic MPC, share of oracle headroom recovered, jerk RMS, mean gap, unsafe windows.

## Reporting
All numbers are reported whether or not they agree with the original (SAC: -0.26% vs MPC const on the legacy plant; +0.06% vs MPC const at eta 0.65 with the legacy-trained policy, 7 unsafe windows).
If the verdict changes, the paper's controller section, abstract and limitations are rewritten to match. Outputs in results/eta065_rerun/sac_*.

## Companion measurement (not a hypothesis test)
Single-process, otherwise idle machine: wall-clock time of every MPC QP solve call (const-speed and oracle previews) over 30 evenly spaced test windows; reported as mean, median, 95th and 99th percentile, max,
with the CPU model. This replaces the rough 83 ms figure from the parallel run.
