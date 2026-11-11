# S1b + S2 fixed-metric report (Sonnet, per admin/indian_prereg_s1b.md, rules written before runs)
Code: python/s1b_run.py, s1b_summary.py, scenarios.py (apply_caps), mpc_acc.py (preview 'constmap', cutin_metrics), baselines.py (cutin_metrics). Tables: results/gate3b/s1b_summary.txt, s2_fixed_summary.txt. Old S2 CSVs archived in results/gate3b/s2_v1_archive/. No weights changed. Design check on 3 TUNE windows passed first (no collision, crossing excess 0.00).

## S1b (42 city-speed test windows, mean lead speed <=20 m/s; all arms respect breakers)
| Arm | Median Wh/km | vs ACC-map | unsafe windows | mean crossing excess (m/s) |
|---|---|---|---|---|
| ACC-map | 139.74 | -- | 0 | 1.22 |
| CACC-map (V2V) | 140.95 | -1.69% (p .011) | 19 | 1.47 |
| MPC-map (old, lead const) | 150.44 | +6.57% | 2 | 0.00 |
| **MPC-map-L** | **127.22** | **-7.40% (p<.0001)** | **0** | **0.00** |
| ORACLE-map | 115.99 | -17.80% | 0 | 0.00 |
PRIMARY RULE: **PASS** (needed <=-1.0%, excess <=1.52, unsafe <=0, p<.05; got -7.40%, 0.00, 0, p<.0001). MPC-map-L beats ACC-map in 38/42 windows. Hits the 4.2 m/s breaker speed every time.

## S2 cut-ins with fixed safety metrics (45 windows; shortfall excluding 5 s after cut-in)
| Arm | Wh/km vs ACC | unsafe (post) | recovery s | not recovered |
|---|---|---|---|---|
| ACC | 187.97 | 0 | 2.47 | 1 |
| CACC | -0.76% | 9 | 2.44 | 1 |
| SMC | -12.13% | 34 | 8.08 | 24 |
| MPC-const | +0.48% (ns) | 1 | 2.51 | 1 |
| MPC-V2V-ML | -1.11% (ns, p .06) | 2 | 2.48 | 1 |
| Oracle | -11.21% | 0 | 2.16 | 1 |
SMC's apparent -12% is NOT a win: it does not re-open the gap after a cut-in (34 unsafe, 24 events unrecovered). Controllers that react safely (ACC, MPC, oracle) recover in ~2.5 s.

## Honest reading
1. Causal MPC wins in Indian breaker traffic only when it knows the road map AND models that the lead obeys the same map (-7.4% vs a fair map-aware ACC, 0 unsafe, 0 speed excess). Using the map alone is worse than ACC (+6.6%).
2. This is a second pre-registration written after diagnosing the first failure (S1). Both must be reported in the paper (S1 FAIL, S1b PASS). Windows: city-speed only (mean <=20 m/s; 30 of 42 are ZalaZone fold 3 data, so the claim is for low-speed/urban-like driving, not highway).
3. Oracle headroom remains (-17.8%): better lead prediction would add much more. V2V-ML still gives no reliable gain (S2: -1.1%, p .06).
4. Limits: breakers/cut-ins/hard braking are stylized injections into real lead trajectories, not recorded Indian traffic; single plant eta .65; MPC-map-L's lead-slowing model is hand-built (not learned).
