# R0 — realistic regen re-run (Sonnet, 2026-11-09) — for Opus review

Change: mpc_acc.pb_kw(pw, v): regen = ETA * R1 * fade(v) * max(pw, -35 kW); fade = min(1, v/3 m/s); excess braking = friction (lost). MPC internal model uses the same slope/cap/fade (3-piece convex max; per-step slope from the SQP reference speed). `set_regen(None)` restores the legacy plant (backup python/mpc_acc_legacy_backup.py; legacy MPC const reproduces stored result within 0.02%). The MPC hard constraint pw >= -35 kW was REMOVED (friction brakes can supply the rest); a limits unchanged. Everything else frozen (weights, ACC k1 .4/k2 1.0, SMC .3/.5/2.0, SAC ckpt 137034 trained under the legacy plant, evaluated only). No retuning.
Coverage: eta 0.65 = all 160 test windows (python/gate3b_r0.py, python/r0_summary.py, results/gate3b/r0_summary_eta0.65.txt). eta 0.8 and 1.0 = every 3rd test window (54, STRIDE=3) because the regen-aware MPC runs ~2x slower (~3.5 s/sim). eta 1.0 here still has cap+fade (not the legacy plant).

## Main case eta 0.65, 160 windows (Wh/km KE-adj)
| controller | Wh/km | median vs MPC-const | gap | unsafe (max shortfall) |
|---|---|---|---|---|
| ACC | 142.84 | -0.47% | 39.6 | 0 |
| SMC | 143.86 | -0.01% | 39.5 | 0 |
| SAC | 144.50 | +0.06% | 41.2 | 7 (1.5 m) |
| MPC const-speed | 144.10 | 0 | 32.0 | 0 (0.2 m) |
| MPC ORACLE (non-causal) | 137.12 | **-3.67%** | 32.6 | 0 (0.1 m) |
Oracle vs ACC: median -2.83%, mean -4.36% (p 3e-24); MPC-const vs ACC: +0.47% (ACC still ahead). By tercile, oracle vs ACC: calm -0.37%, medium -3.50%, **dynamic -7.82%**.
Sensitivity (54 windows): eta 0.8: oracle vs ACC -1.99%, dynamic -6.28%; eta 1.0 (cap+fade): -1.14%, dynamic -4.46%. MPC-const vs ACC stays +0.47..0.50% at every eta.

## Readings
1. Opus's hypothesis confirmed: the preview headroom grows as regen gets more realistic (oracle vs MPC-const: -1.7% at eta 1.0 -> -2.75% at 0.8 -> -3.67% at 0.65; dynamic tercile vs ACC -7.8% at 0.65).
2. Causal MPC without preview still does not beat tuned ACC (+0.47%), independent of eta.
3. The 3 large-shortfall windows (up to 14 m) are gone for MPC-const at every eta (max 0.2 m): consistent with them being caused by the removed -35 kW wheel-power hard constraint, not only by the preview. (Not separately proven; the legacy-with-constraint-removed run was not isolated.) So the 'safety layer' step may be less needed; check before adding.
4. SAC was trained under the legacy plant (free braking); retraining under eta .65 may change it - not done.
Caveats: eta/cap/fade are assumed, not identified (BMW i3 regen limits not verified; sensitivity shown); eta 0.8/1.0 are a 54-window subset; flat road, constant aux; MPC solver warm-start slower now.
