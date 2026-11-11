# Pre-registration S1b (Opus, written 2026-11-09 before any run). Second pre-registration after the failed S1; both are reported.
Windows: ALL role=test rows of results/v2v_windows.csv with mean recorded lead speed <= 20 m/s (42 windows; folds 3:30, 1:7, 2:5). Fixed before running. Odd rows were not used in S1, so this is not a subset chosen by S1 outcome.
Breakers: same generator and seeds rule (7000+100*1+row) as S1. The lead profile is unchanged. NEW: the same breaker cap profile is also applied to the predecessor speed vp (positioned 5+1.5*vp[0] m ahead of the lead at t=0, same rate limiter) before the V2V channel.
Plant eta .65; all MPC weights frozen; w_cap 20 unchanged.
Arms (all breaker-respecting): ACC-map (unchanged), MPC-map-L = MPC-map whose lead preview is min(const-speed preview, lead breaker cap profile computed from the lead's estimated position xe+d, same vb 4.2, a_dec 2.0, a_acc 1.0); ORACLE-map = MPC-map with oracle preview (ceiling); CACC-map = CACC with the ACC-map speed limit (min rule). Also report MPC-map (old, unchanged) on these windows.
Implementation must be verified on 3 TUNE-role windows (design check only: no collision, crossing excess < 0.5) before test runs; no weight changes allowed. If the design check fails, stop and report; do not iterate.
PRIMARY RULE (S1b): MPC-map-L median Wh/km >= 1.0% below ACC-map AND mean crossing excess <= ACC-map's + 0.3 m/s AND unsafe windows (>0.5 m) <= ACC-map's AND Wilcoxon p < 0.05. Report PASS/FAIL exactly.
Secondary (no pass/fail): ORACLE-map vs ACC-map (headroom given breaker respect); CACC-map vs ACC-map.

# S2 metric fix (re-run S2 all arms, same windows/seeds; old S2 table kept)
New safety metrics: (a) shortfall_post = max shortfall excluding the 5 s after each cut-in; (b) recovery time = time from each cut-in until shortfall <= 0.5 m (mean, and count of events not recovered within 15 s); (c) min gap. "Unsafe window" for S2 = shortfall_post > 0.5 m. Energy comparison unchanged. No pass/fail.
