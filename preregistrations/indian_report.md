# Indian mixed-traffic scenarios: report (Sonnet, run exactly per admin/indian_prereg.md)
Setup: 45 test windows (even-position rows of role=test), eta .65, seeds 7000+100*scenario+row, all MPC weights frozen. Code: python/scenarios.py, scen_run.py, scen_summary.py; table: results/gate3b/scenarios_summary.txt.
Implementation notes (not outcome tuning): added a soft speed cap at predicted breaker crossings (w_cap 20) and ACC-map baseline (sqrt-profile approach). First smoke test used a lead that ignored breakers (not meaningful); design check on 3 TUNE windows only showed the cap works (crossing excess 0.0 vs 1-2 m/s). No weights changed afterward.

## Results (median Wh/km, change vs plain ACC)
| Scenario | ACC | CACC | SMC | MPC-const | MPC-V2V-ML | Oracle |
|---|---|---|---|---|---|---|
| S1 breakers | 234.97 | -0.33% | +4.87% | +7.20% | +7.32% | -26.71% |
| S2 cut-ins | 187.97 | -0.76% | -12.13% | +0.48% | -1.11% | -11.21% |
| S3 hard brake | 190.42 | -0.71% | +1.52% | +2.04% | +1.60% | -18.34% |
S1 map-aware: ACC-map +1.14% vs ACC, MPC-map +11.96% vs ACC.
Crossing speed excess over 4.2 m/s (S1, mean m/s): ACC 2.73, ACC-map 1.39, MPC-const 2.25, MPC-map 0.61. Unsafe windows (>0.5 m shortfall) S1: ACC-map 0, MPC-map 14.

## PRIMARY RULE: FAIL
MPC-map vs ACC-map: median +11.84% (needed <= -1.0%), unsafe 14 vs 0, p<0.0001 in the wrong direction. Only the crossing-excess criterion passed (0.61 vs 1.39).

## Honest reading
1. The causal battery-aware MPC does not beat a tuned ACC in Indian-style disturbances either; the map-aware MPC meets the breaker speed much better but pays ~12% energy and breaks the gap rule in 14 windows (cap fights gap keeping; 15 s horizon, limited braking).
2. The oracle is 11-27% better in every scenario: the value is in knowing what the lead will do, not in the optimiser. V2V-ML forecasts do not capture it (they give -1.1% in S2 only, p=.06).
3. S2 caveat: cut-in gaps (8-15 m) are below the safe gap in nearly all arms (36-43 of 45 unsafe), so safety columns do not discriminate there; SMC's -12% in S2 mainly reflects it tolerating shorter gaps / less braking, so do not call it better without the safety caveat. S2/S3 had no pass/fail rule.
4. Paper use: report as a negative-but-informative scenario study (causal MPC ~ ACC; preview value large only with non-causal information). Do not claim an Indian-traffic advantage.
Open question for Opus: is the MPC-map result an implementation weakness (soft cap, shortfall) worth one pre-declared fix, or accept the negative result?
