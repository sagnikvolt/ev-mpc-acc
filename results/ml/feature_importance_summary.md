# Which inputs govern the battery power (BMW i3, 68 trips, 5-fold trip-wise CV)

Permutation importance on HELD-OUT trips: increase in power RMSE (kW) when one input group is shuffled; mean over 5 folds (3 repeats for LightGBM, 2 repeats on 5,000 test rows for SVR/KNN). Bigger = the model relies on it more. "Physics wheel power (Pw)" exists only in the residual models.

| Input group | LightGBM pure (no physics) | LightGBM + physics | SVR + physics | KNN + physics |
|---|---|---|---|---|
| Acceleration (a, av, lags) | 14.237 | 0.348 | 0.288 | 0.261 |
| Speed (v, v2, lags) | 4.672 | 0.227 | 0.243 | 0.154 |
| Cabin heating | 0.379 | 0.181 | 0.191 | 0.202 |
| Road grade | 0.269 | 0.144 | 0.090 | 0.066 |
| Ambient temperature | 0.263 | 0.107 | 0.259 | 0.040 |
| Battery temperature | 0.182 | 0.114 | 0.126 | 0.039 |
| Air-conditioning | 0.071 | 0.068 | 0.077 | 0.031 |
| State of charge | 0.054 | 0.057 | 0.010 | -0.007 |
| Physics wheel power (Pw) | - | 0.108 | 0.066 | 0.068 |

Base RMSE (kW, mean over folds): lgbm_pure 4.14, lgbm_resid 4.03, svr_resid 4.12, knn_resid 4.20

LightGBM pure, per-fold ranking stability (rank of each group in each fold, 1 = most important):
fold                        0  1  2  3  4
group                                    
Acceleration (a, av, lags)  1  1  1  1  1
Speed (v, v2, lags)         2  2  2  2  2
Cabin heating               3  6  3  4  3
Road grade                  6  3  5  3  5
Ambient temperature         4  4  4  5  4
Battery temperature         5  5  6  7  7
Air-conditioning            7  8  8  6  6
State of charge             8  7  7  8  8

# Learning curve: how much data is enough (LightGBM, trip-wise subsets of the training trips, 5 folds x 3 draws)

| Training samples (2 Hz) | Training trips | RMSE kW, physics + LightGBM | RMSE kW, pure LightGBM |
|---|---|---|---|
| requested 10000 (11509) | 4 | 4.60 ± 0.46 | 5.10 ± 0.73 |
| requested 25000 (26809) | 9 | 4.43 ± 0.47 | 4.55 ± 0.42 |
| requested 50000 (52486) | 18 | 4.28 ± 0.51 | 4.39 ± 0.40 |
| requested 100000 (102005) | 34 | 4.13 ± 0.48 | 4.24 ± 0.38 |
| requested full (161482) | 54 | 4.03 ± 0.37 | 4.14 ± 0.32 |

# Dataset sizes

| Dataset | Unit | Count | Trips / segments | Hours | Use |
|---|---|---|---|---|---|
| BMW i3 (TU Munich, measured battery power) - energy-model data | samples at 2 Hz | 201,852 | 68 | 28.0 | physics fit, residual ML, feature importance, learning curve |
| Renault Zoe (public trip data) | samples at 1 Hz | 400,216 | 471 | 111.2 | second-vehicle dataset (cleaned, trip-wise split); not used in the tables here |
| OpenACC (JRC real car-following, 10 Hz) | samples at 10 Hz | 2,796,734 | 961 | 77.7 | lead-vehicle forecasting, controller tests |
| OpenACC 120 s windows, folds 0-4 (all) | windows | 203 | 70 | 6.8 | closed-loop experiments (43 tune, 160 test) |
| V2V 120 s windows (30 tune, 90 test) from 961 segments | windows | 120 | 961 | 4.0 | V2V forecasting and closed-loop check |

## Notes (read before quoting)
- Acceleration then speed govern the power in every model; in the pure model they dominate (14.2 and 4.7 kW). With the physics model supplying the bulk of the power, the residual models rely on small, similar-sized contributions from acceleration, speed, cabin heating, grade, temperatures. Cabin heating is the largest non-driving input.
- Ranks of the top two groups are identical in all 5 folds; ranks below them move by 1-3 places, so only "acceleration > speed > the rest" is a firm ordering.
- SVR/KNN importance uses 5,000 held-out rows x 2 repeats per fold (cost). BiLSTM importance was not computed (same inputs, window-based permutation not run).
- Grouped permutation, not SHAP; correlated inputs share credit inside a group by construction.
- Learning curve: trips are added whole (held-out trips never in training). Error falls from 4.60 kW (11.5k samples, 4 trips) to 4.03 kW (161k, 54 trips) and is still falling slowly (4.13 at 102k), so the data is at the "lakhs" range Sir asked for but more trips would still help slightly. With about 11.5k samples the error is 14% above the full-data error; with about 52k samples it is 6% above.
