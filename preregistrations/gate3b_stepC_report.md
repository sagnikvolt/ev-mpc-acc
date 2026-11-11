# Gate 3B step C — oracle-preview vs constant-speed-preview MPC (Sonnet, for Opus review)

**Setup.** python/openacc_prep.py -> 961 (lead,follower) segments from 6 road/track campaigns (cache/openacc_segs.npy; results/openacc_summary.csv). One-vehicle campaign skipped (no lead). python/freeze_params.py -> frozen BMW i3 model fitted on all 68 trips (results/bmw_i3_params_final.npy: Crr .0102, CdA .466, traction 1.43 P, regen 1.0 P (at cap), aux 110 W). python/mpc_acc.py: battery-aware MPC (cvxpy/OSQP, N=30, dt=0.5 s, 15 s horizon, 3 SQP passes). Closed loop on 78 non-overlapping 120 s windows (pair-0, human-driven lead, mean speed >5 m/s, std >1 m/s) from python/gate3b_oracle.py.

**Result (results/gate3b/oracle_summary.txt).**
- Oracle (true 15 s lead preview) vs constant-speed preview: Wh/km mean -3.3%, median -2.2%, pooled -3.1%; better in 72/78 windows; Wilcoxon p ~ 1e-13. End-speed-adjusted: median -2.1%.
- Safety: const-preview had 5 windows with soft-margin shortfall >0.5 m (max 28 m), 3 windows with min gap <3 m. Oracle had 0 (max shortfall 0.1 m). Comfort: jerk RMS 0.26 vs 0.32.
- By campaign (median adj): Casale -1.7, Cherasco -1.7, JRC low-speed -3.1, Vicolungo -4.2, ZalaZone -2.6.

**Gate rule (<2% => go Plan C): median 2.1-2.2% = marginal pass; mean 3.1-3.3% clear pass.** This is a CEILING: real forecasts will recover only part of it.

**Caveats / honesty.** (1) Controller weights/gap band untuned, same for both arms. (2) Flat road, aux = constant 110 W (no HVAC), no cut-ins/potholes yet. (3) R2*P^2 regen term dropped to keep a QP (<=0.4 kW). (4) Constant-speed baseline is the Fiorillo-style assumption; conventional ACC, SMC, SAC not run yet. (5) 120 s windows; lead = human drivers in platoon experiments (ZalaZone is a test track). (6) Windows are not day-split yet; only matters for the forecasters.

**Needs Opus decision:** is ~3% ceiling + safety benefit enough to proceed with forecasting (step A) or should the paper lean on Plan C / a different main claim?
