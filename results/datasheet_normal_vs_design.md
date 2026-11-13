# Datasheet: normal car vs our design car (battery model: BMW i3 physics, regen efficiency 0.65, 35 kW cap; KE-adjusted Wh/km; day-held-out test windows)

Normal car = the follower vehicle actually recorded in OpenACC (commercial ACC or human), its speed trace replayed through the same battery model, 1 s smoothing (pre-registered). Gap = mean following gap in metres (recorded gap missing in 40 windows, mostly human-driven). Jerk = RMS, m/s^3. Unsafe = windows with gap shortfall > 0.5 m against the safe-gap rule 4 + 0.7 v (not computed for recorded cars: different gap definition). Wh/km means are over the windows in the panel; the % column is the paired per-window median change versus the recorded follower (negative = uses less).

## All 160 test windows (n = 160)

| Vehicle / controller | Mean Wh/km | Change vs recorded car | Mean gap (m) | Jerk RMS | Unsafe windows | Compute per decision |
|---|---|---|---|---|---|---|
| Normal car: recorded real follower (replayed through the same battery model) | 150.1 | 0 (reference) | 34.8 | 0.23 | n/a | - |
| Tuned ACC (simulated, conventional controller) | 142.8 | -3.01% (median) | 39.6 | 0.21 | 0 | not timed (simple formula / small network) |
| Sliding-mode controller (SMC) | 143.9 | -2.79% (median) | 39.5 | 0.24 | 0 | not timed (simple formula / small network) |
| Plain SAC (RL, seed 42) | 144.3 | -2.28% (median) | 45.9 | 0.42 | 2 | not timed (simple formula / small network) |
| Our design car: battery-aware MPC (causal) | 144.1 | -2.47% (median) | 32.0 | 0.27 | 0 | median 29 ms, p99 110 ms |
| MPC oracle (non-causal ceiling) | 137.1 | -5.99% (median) | 32.6 | 0.22 | 0 | n/a (needs the lead car's future) |
| Safe SAC (RL + safety filter, seed 42) | 144.4 | -2.50% (median) | 43.2 | 0.27 | 0 | not timed (simple formula / small network) |

## 79 windows with a commercial ACC follower (n = 79)

| Vehicle / controller | Mean Wh/km | Change vs recorded car | Mean gap (m) | Jerk RMS | Unsafe windows | Compute per decision |
|---|---|---|---|---|---|---|
| Normal car: recorded real follower (replayed through the same battery model) | 176.6 | 0 (reference) | 40.0 | 0.21 | n/a | - |
| Tuned ACC (simulated, conventional controller) | 169.3 | -2.82% (median) | 50.0 | 0.20 | 0 | not timed (simple formula / small network) |
| Sliding-mode controller (SMC) | 170.1 | -2.27% (median) | 50.0 | 0.23 | 0 | not timed (simple formula / small network) |
| Plain SAC (RL, seed 42) | 168.7 | -2.86% (median) | 59.1 | 0.40 | 0 | not timed (simple formula / small network) |
| Our design car: battery-aware MPC (causal) | 171.2 | -1.89% (median) | 41.3 | 0.24 | 0 | median 29 ms, p99 110 ms |
| MPC oracle (non-causal ceiling) | 164.0 | -5.49% (median) | 41.8 | 0.19 | 0 | n/a (needs the lead car's future) |
| Safe SAC (RL + safety filter, seed 42) | 169.5 | -2.80% (median) | 57.8 | 0.25 | 0 | not timed (simple formula / small network) |

Notes: (1) MPC compute time measured on an i5-12500H, one core (30 windows, 12,540 solves); the control step is 500 ms. ACC, SMC and SAC decision times were not timed; they are a simple formula or a small network. (2) The oracle knows the lead car's future and is a ceiling, not a deployable car. (3) The tuned ACC saves slightly more than the causal MPC against the recorded cars; the MPC's advantage appears only with preview (oracle). (4) Gap matters: in the commercial-ACC panel the tuned ACC and SAC follow at about 50-59 m versus 40 m for the recorded cars and 41 m for our MPC; a longer gap lets a car drive more smoothly, so part of their saving is the larger gap. The MPC compares with the recorded cars at the same gap. (5) Mean Wh/km differs between panels because the windows differ.