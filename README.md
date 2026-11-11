# Battery-aware MPC adaptive cruise control for EVs: open data, code and pre-registrations

Code, pre-registered test rules, result tables and derived data behind the paper on battery-aware model-predictive adaptive cruise control (ACC) for battery-electric vehicles (BMW i3 physics, real lead-vehicle trajectories from OpenACC). Status: **DRAFT for Zenodo upload. The DOI is assigned only on upload; the author list is pending (see TODO in CITATION.cff and zenodo_metadata.json).**

## What is in this package
| Folder | Content |
|---|---|
| `code/python/` | All Python code that produced the results (plant + frozen BMW i3 model `mpc_acc.py`, controllers `baselines.py`, forecasters `fc_*.py`/`v2v_*.py`, scenario generators `scenarios.py`, run and summary scripts, `make_package.py`). |
| `code/matlab/` | MATLAB R2024b port of plant + ACC + MPC (`verify_main.m`) and the comparison script. Needs the Optimization Toolbox (`quadprog`). |
| `preregistrations/` | Test rules written **before** the runs (`*_prereg*.md`) and the run reports (`*_report.md`). The failed first Indian-scenario test (S1) and the follow-up (S1b, a second pre-registration written after diagnosing S1) are both included. |
| `results/` | Result tables (CSV/TXT) for every experiment, incl. `gate3b/` (controller benchmarks, forecasters, V2V, scenarios S1/S1b/S2/S3) and the MATLAB verification. |
| `derived_data/` | Our own derived files: fold assignment, evaluation windows, fitted BMW i3 parameters, and the 10 Hz lead-vehicle speed windows used in the simulations (see below). |
| `code/paper_figures/` | `make_figures.py`: builds the two paper figures (outputs in `results/figures/`). |
| `quickstart_check.py` | **Start here.** Reruns the conventional ACC and the oracle MPC on a few test windows using only files in this package and checks them against the stored results (about 30 s). |
| `models/` | Trained SAC checkpoint used in the benchmark (`sac_ckpt_137034.zip`, Stable-Baselines3). |
| `DATA_SOURCES.md` | Third-party datasets, licences, citations and download instructions. **Raw third-party data is not included.** |

## Headline results (see the reports for all numbers, tests and limits)
- A causal battery-aware MPC is **not** better than a well-tuned conventional ACC on real OpenACC lead trajectories (differences below 1%); the value is in knowing the lead vehicle's future (non-causal oracle ceiling).
- Indian-style speed-breaker scenario, first pre-registered test (S1): **FAIL** (MPC with map but constant-speed lead assumption +11.8% energy vs a map-aware ACC, 14 windows with gap shortfall).
- Second pre-registration (S1b, 42 urban-speed windows): MPC that also models the lead slowing for breakers used **7.4% less energy** than the map-aware ACC (p<0.0001, no gap violations, no overspeed at breakers); -6.6% and -6.8% when the real lead's braking rule differs from the model's (robustness test). Disturbances are stylised injections into real trajectories, not recorded Indian traffic.
- Regeneration power cap raised from 35 to 50 kW (sensitivity run, rules in `preregistrations/sens_cap50_prereg.md`, results in `results/sens_cap50/`): oracle saving versus ACC changes from -2.83% to -2.65% (median, 160 windows); S1b saving of MPC-map-L versus map-aware ACC from -7.40% to -6.97% (38 of 42 windows better, no unsafe window).
- MATLAB R2024b re-implementation reproduces Python to 0.000% (ACC) and 0.159% max / -0.008% mean (MPC) on 10 fixed windows.
- A learned V2V forecaster improves forecast RMSE but gives no reliable closed-loop energy gain.

## Reproducing
Environment used: Python 3.10.12, numpy 2.2.6, pandas 2.3.3, scipy 1.15.3, cvxpy 1.7.5, osqp 1.1.3, lightgbm 4.7.0, scikit-learn 1.7.2, stable-baselines3 2.9.2, torch 2.14 (CPU), joblib 1.6.0 (`requirements.txt`); MATLAB R2024b Update 5. All scripts define `ROOT` near the top (default `~/mnt/EV Project`); set it to your project folder (which must contain `data/raw/...` as in DATA_SOURCES.md, plus `cache/` and `results/`, created by the scripts).
1. Download the datasets (DATA_SOURCES.md) into `data/raw/`.
2. `openacc_prep.py` (segments) -> `folds.py` (day-based folds) -> `freeze_params.py` / `fit_bmw_i3*.py` (BMW i3 model; the final parameters are in `derived_data/bmw_i3_params_final.csv`).
3. Forecasters: `fc_data.py`, `fc_bench.py`, `fc_lstm.py`, `fc_score.py`; V2V: `v2v_prep.py`, `v2v_fc.py`, `v2v_lgbm.py`.
4. Closed-loop experiments: `gate3b_*.py`, `v2v_run.py`, `sac_train.py`/`gate3b_sac_test.py`, scenarios `scen_run.py` and `s1b_run.py`; summaries `*_summary.py`.
5. MATLAB: in MATLAB R2024b, `cd` to a folder containing `verify_main.m` and `data/` (copy `derived_data/matlab_verification_windows` as `data`), run `verify_main`, then `python compare.py`.
Quick check without re-running everything: `python3 quickstart_check.py`. The lead-speed windows are in `derived_data/*.npz` and the simulator is `code/python/mpc_acc.py` (`simulate`) and `baselines.py`.
Run times: one MPC window about 4-5 s; the complete experiment set takes many hours on 2 CPU cores.

## Known limits
- Stylised disturbances; single plant efficiency setting for the main results (regen efficiency 0.65; 0.8 and 1.0 reported as sensitivity).
- The MATLAB port covers ACC and MPC with constant-speed preview only.
- Windows from the ZalaZone test track dominate the low-speed (urban-like) test set.
- Cached trained forecaster models (LightGBM) are not included; they are re-created by `v2v_lgbm.py` / `v2v_save_models.py`.

## About the pre-registration dates
The rule files in `preregistrations/` were written before the corresponding runs (see the file modification times in the original project folder and the order of the reports). A public repository can only prove dates from its own first commit onwards, so the repository history documents the release, not the original writing time. Anyone re-running the study should commit their own rule files before running.

## Licence
Code: MIT (`LICENSE`). Our own result tables and derived files: CC BY 4.0. Third-party data keeps its own terms (DATA_SOURCES.md). **TODO before upload: confirm licences and author list with all co-authors.**

## Live demo (optional)
`aws_demo/` holds a small web demo (AWS Lambda container + static page) that runs the same model, ACC and MPC on a speed trace you provide. See `aws_demo/DEPLOY.md`. `python3 aws_demo/test_local.py` checks it against the stored paper results without AWS. A hosted copy is at https://ev.sagniknath.in (research demo only).
