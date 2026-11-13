# Pre-registration: safe RL (SAC + control-barrier safety filter) at eta 0.65

Written 2026-11-13 (Opus 5.5 + Roni) BEFORE any filtered-SAC run. Answers Sir's "why just MPC": the strongest current alternative family is
safe RL (learned policy + safety filter). Plain SAC at eta 0.65 is already done (results/eta065_rerun/summary_sac.txt: seed 42 not better than ACC,
unsafe windows 2/6/37 for seeds 42/43/44). This test asks: if we make SAC safe, does it then beat ACC on energy?

## Safety filter (fixed)
Barrier h = d - (4.0 + 0.7 v)  (same safe-gap rule used for "unsafe" everywhere). Control step dt = 0.5 s, accel a constant over the step.
Worst-case lead over the step: lead speed vl with its current decel kept, al_neg = min(al, 0), where al is the same 1 s lead-accel estimate the SAC observation uses.
Discrete CBF condition: h(next) >= (1 - gamma) * h(now). This is linear in a, giving an upper bound
  a_cbf = [ d + dt*(vl + 0.5*al_neg*dt - v) - 4.0 - 0.7 v - (1 - gamma)*h ] / (0.5*dt^2 + 0.7*dt)
Executed accel a = clip(min(a_rl, a_cbf), -3.5, 2.0). If a_cbf < -3.5 the step is logged as "filter infeasible" (full braking is applied).
The filter only ever lowers acceleration; it never adds energy-saving knowledge.

## gamma (one choice, made on fold 0 only)
Candidates {0.1, 0.3, 0.5}. Chosen with the existing eta-0.65 SAC seed 42 checkpoint + filter at evaluation only, on the 43 fold-0 windows:
lowest mean KE-adjusted Wh/km among candidates with 0 unsafe windows; ties/none safe -> smallest gamma. Then frozen for everything below.

## Arms
- B1 (cheap, secondary): the three existing eta-0.65 SAC models (seeds 42/43/44, already-selected checkpoints) + filter at test time only. No retraining.
- B2 (PRIMARY): SAC retrained with the filter inside the environment (executed = filtered action; reward computed on the executed action;
  extra reward term -0.1*|a_rl - a_exec| so the policy learns to need the filter less). Everything else identical to rerun_sac_eta065.py:
  [128,128], lr 3e-4, buffer 200k, batch 256, learning_starts 2000, gamma_rl .98, tau .01, 3 seeds (42 primary, 43, 44), 10 chunks x 20k steps,
  checkpoint chosen by fold-0 score (Wh/km + 2 * mean shortfall), test folds 1-4 (160 windows) run ONCE.
- References (existing results, not re-run): tuned ACC r0_0.65_acc, MPC const, MPC oracle, plain SAC eta 0.65; and recorded followers (results/normal_car).

## Pass rule (primary, one test: B2 seed 42 vs simulated ACC, 160 test windows)
median per-window change in KE-adjusted Wh/km <= -1.0%, AND Wilcoxon p < 0.05, AND 0 unsafe windows (shortfall > 0.5 m).
Seeds 43/44 and B1 are reported with the same numbers and carry no pass rule. "Failed to learn" if every checkpoint's fold-0 mean shortfall > 5 m.

## Also reported
median % vs MPC const and vs recorded ACC-driven followers (step A windows); unsafe count; max shortfall; filter intervention rate
(% of steps with a_cbf < a_rl); filter-infeasible steps; mean gap; jerk RMS; training wall time.

## Honesty notes
- If safe SAC wins only by keeping a longer gap, that is said next to the result (mean gaps reported).
- If it fails, it is reported as failed; gamma, reward and hyperparameters are not changed after seeing test results.
- Output: results/safe_rl/ ; scripts python/safe_rl_gamma.py (fold-0 gamma pick + B1), python/safe_rl_train.py (B2), python/summary_safe_rl.py ; run_safe_rl.bat.
