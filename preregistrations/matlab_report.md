# MATLAB R2024b verification report (Sonnet, per admin/matlab_prereg.md, written before the run)
Machine: a Windows laptop, MATLAB 24.2 (R2024b) Update 5, Optimization Toolbox (quadprog, interior-point-convex). Run by driving the MATLAB window; code: matlab/verify/verify_main.m; data matlab/verify/data/; results matlab/verify/matlab_results.csv; comparison matlab/verify/compare.py -> results/gate3b/matlab_verification.txt.
Scope: independent port of the frozen BMW i3 physics + realistic regen (eta .65), conventional ACC, and battery-aware MPC (N=30, dt .5 s, 3 SQP passes, same weights/constraints) with constant-speed preview, on 10 fixed test windows (rows 0,9,...,81 of results/v2v_windows.csv).
## Result: PASS
| Arm | Max per-window diff vs Python | Mean diff | Tolerance | |
|---|---|---|---|---|
| ACC | 0.000% | 0.000% | 0.1% | PASS |
| MPC-const | 0.159% | -0.008% | 1.0% / 0.5% | PASS |
dmin identical to 0.01 m in all windows; MATLAB solver failures 0; MPC-vs-ACC ranking per window identical (e.g. +0.9/+0.9, +3.8/+4.0). Jerk RMS differs by up to 0.03 m/s^3 (solver tie-breaking in flat parts of the cost); energy and gap metrics are not affected.
## Limits (state in the paper)
- Verified: ACC and MPC-const (the core model and controller). NOT ported/verified in MATLAB: breaker map-aware MPC (S1b), V2V-ML, SAC, SMC, cut-in scenarios. Claim wording: "the core simulator and MPC were independently re-implemented in MATLAB R2024b and reproduce the Python results to within 0.16% on 10 windows".
- Same author wrote both ports; this checks implementation consistency (solver, indexing), not the physical model against measured car data.
