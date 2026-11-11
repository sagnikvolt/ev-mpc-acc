# Sensitivity: regeneration power cap 50 kW (written 2026-11-09 before running)
Why: BMW i3 press material quotes recuperation up to ~50 kW; the paper assumes 35 kW. Reviewers will ask.
Change: ONLY CAP = 50 kW (eta 0.65, fade 3 m/s). All weights, ACC/SMC settings, windows and seeds frozen. No re-tuning.
Runs: (1) ACC and MPC-oracle on the 160 test windows (same as R0); (2) S1b: ACC-map and MPC-map-L on the same 42 windows and seeds.
Report (no pass/fail rule, sensitivity only): median % change oracle vs ACC (all, dynamic third by the same lead-acceleration terciles), S1b median MPC-map-L vs ACC-map, unsafe windows, Wilcoxon p.
Reading rule: if the S1b saving stays >= 1.0% with no unsafe window, the S1b claim is unchanged; otherwise it is reported as cap-sensitive.
Script: python/sens_cap50.py. Output: results/sens_cap50/.
