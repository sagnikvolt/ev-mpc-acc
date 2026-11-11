# Controller benchmark (Sonnet, 2026-11-08) — for Opus review. IMPORTANT NEGATIVE FINDING.

Code: python/baselines.py (common simulator: same plant, 0.5 s control, a in [-3.5,2], jerk<=3, same metrics), python/gate3b_base.py (ACC/SMC tune+test), python/sac_env.py + sac_train.py + gate3b_sac_test.py (SB3 SAC), results/gate3b/controllers_test_summary.txt, acc_gap_fairness.txt.
Tuning: ACC (12 configs) and SMC (12 configs) tuned on fold 0 with the same rule as the MPC (Wh/km + 2*shortfall, jerk <= 0.487); chosen ACC k1=.4, k2=1.0 (target gap 5+1.5 v); SMC lambda .3, K .5, phi 2.
SAC: trained ONLY on folds 0 and -1 segments (test days unseen), 201k steps (~20 min CPU), first attempt (v1) learned to crash early (collision penalty too small) and was discarded (cache/sac_v1_failed); v2 reward fixed. Checkpoint chosen by fold-0 score only (137,034 steps), then run once on test.

## Test, 160 held-out windows (energy Wh/km KE-adjusted)
| controller | Wh/km | median vs MPC(const) | mean gap | unsafe windows (max shortfall) | jerk |
|---|---|---|---|---|---|
| Conventional ACC (PD) | 138.20 | **-0.49%** | 39.6 m | 0 | 0.21 |
| SMC | 138.68 | -0.29% | 39.5 | 0 | 0.24 |
| SAC | 138.96 | -0.26% | 41.2 | 7 (1.5 m) | 0.26 |
| MPC, const-speed preview | 139.35 | 0 | 32.1 | 3 (14.1 m) | 0.27 |
| MPC, const-accel preview | 140.08 | +0.28% | 32.3 | 3 (8.2 m) | 0.43 |
| MPC, oracle preview (non-causal) | 135.95 | -1.78% | 32.4 | 0 | 0.22 |
By tercile (median vs MPC const): dynamic: ACC -0.63%, SMC -0.33%, SAC -0.05%, oracle -4.40%.

## Gap-fairness check (the three non-MPC controllers sit ~8 m farther back than the MPC)
ACC with tighter target gaps: 4+1.3 v -> gap 34.0 m, -0.34% vs MPC, 0 unsafe; 3+1.2 v -> gap 30.7 m, -0.28%, 47 windows with <=1.0 m soft-margin shortfall. Even at matched gap, tuned ACC is slightly better than the causal MPC (energy) and safer (no 14 m shortfalls).

## Honest reading
1. A causal battery-aware MPC (const-speed or const-accel preview) does NOT beat a well-tuned conventional ACC or SMC in energy on real OpenACC lead trajectories; differences are small (0.3-0.5% median) but significant. It also has 3 large-shortfall windows (hard lead braking) ACC does not.
2. Only the non-causal oracle-preview MPC wins (-1.3% vs ACC median; dynamic tercile about -3.8% vs ACC, from -4.4% vs MPC and ACC -0.6%).
3. SAC (RL) is no better than plain ACC and is less safe (7 windows) -> RL adds nothing here at this budget.
4. So 'energy-optimal MPC' as the contribution is not supported by the data; the supported statements are: the value of lead preview, the validated physics model, the negative ML result, and that simple ACC is a strong baseline.
Open caveats: MPC cost uses a gap deadband; MPC weights barely matter (tuned grid); flat road; aux constant; MPC tuning could include an energy-vs-gap trade (e.g. narrower band) - not explored; ACC/SMC/SAC use the same energy model, so differences are control-only.
Needs Opus: decide the paper's controller story (reframe around information value / honest benchmark; consider an MPC with gap-band tuned for energy; consider whether SMC/SAC as 'strategies' satisfy sir's 2-3 strategies).
