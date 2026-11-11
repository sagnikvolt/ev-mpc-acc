"""Builds open_data_package/ from the project folder (code, pre-registrations, results, derived data). Raw third-party datasets are NOT copied."""
import os, shutil, glob, hashlib, re, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/EV Project'); OUT = f'{ROOT}/open_data_package'
if os.path.exists(OUT): shutil.move(OUT, f'{ROOT}/_old_package_{int(__import__("time").time())}')   # no rm in this environment
for d in ['code/python', 'code/matlab', 'preregistrations', 'results/gate3b', 'results/ml', 'derived_data', 'models']: os.makedirs(f'{OUT}/{d}')
for f in sorted(glob.glob(f'{ROOT}/python/*.py')):
    b = os.path.basename(f)
    if 'backup' in b: continue
    shutil.copy(f, f'{OUT}/code/python/{b}')
for b in ['verify_main.m', 'compare.py']: shutil.copy(f'{ROOT}/matlab/verify/{b}', f'{OUT}/code/matlab/{b}')
pre = ['indian_prereg.md', 'indian_prereg_s1b.md', 's1b_mismatch_prereg.md', 'matlab_prereg.md']; rep = ['stepA_report.md', 'gate3b_stepC_report.md', 'gate3b_full_report.md', 'r0_report.md', 'r1_report.md', 'controllers_report.md', 'stepV_report.md', 'indian_report.md', 'indian_s1b_report.md', 'matlab_report.md']
for b in pre + rep:
    t = open(f'{ROOT}/admin/{b}').read(); t = t.replace('<hostname>', 'a Windows laptop'); open(f'{OUT}/preregistrations/{b}', 'w').write(t)
for f in glob.glob(f'{ROOT}/results/*.csv') + glob.glob(f'{ROOT}/results/*.md') + glob.glob(f'{ROOT}/results/*.npy'):
    if os.path.basename(f) in ('windows.csv', 'v2v_windows.csv', 'folds.csv', 'seg_folds.csv', 'openacc_summary.csv', 'v2v_candidates.csv'): shutil.copy(f, f'{OUT}/derived_data/' + os.path.basename(f))
    else: shutil.copy(f, f'{OUT}/results/' + os.path.basename(f))
for f in glob.glob(f'{ROOT}/results/gate3b/*') + glob.glob(f'{ROOT}/results/ml/*'):
    if os.path.isfile(f): shutil.copy(f, f'{OUT}/results/' + ('gate3b/' if '/gate3b/' in f else 'ml/') + os.path.basename(f))
shutil.copytree(f'{ROOT}/matlab/verify/data', f'{OUT}/derived_data/matlab_verification_windows')
for b in ['chk_env.txt', 'matlab_results.csv', 'compare_report.txt']: shutil.copy(f'{ROOT}/matlab/verify/{b}', f'{OUT}/results/matlab_{b}')
P = np.load(f'{ROOT}/results/bmw_i3_params_final.npy'); names = ['Crr', 'c1', 'c2', 'r1', 'r2', 'b0_W', 'b1', 'b2', 'b3', 'CdA_m2']
pd.DataFrame({'name': names, 'value': P}).to_csv(f'{OUT}/derived_data/bmw_i3_params_final.csv', index=False)
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv')
np.savez_compressed(f'{OUT}/derived_data/openacc_lead_windows_203.npz', lead_speed_mps=np.array([S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200] for _, r in W.iterrows()], np.float32), window_row=np.arange(len(W)), note='rows align with windows.csv; 10 Hz lead-vehicle speed, 1200 samples (120 s)')
S2 = np.load(f'{ROOT}/cache/v2v_segs.npy', allow_pickle=True); V = pd.read_csv(f'{ROOT}/results/v2v_windows.csv')
np.savez_compressed(f'{OUT}/derived_data/openacc_v2v_windows_120.npz', lead_speed_mps=np.array([S2[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200] for _, r in V.iterrows()], np.float32), predecessor_speed_mps=np.array([S2[int(r.seg)]['vp'][int(r.w0):int(r.w0) + 1200] for _, r in V.iterrows()], np.float32), window_row=np.arange(len(V)), note='rows align with v2v_windows.csv; 10 Hz speed of the lead and of the lead\'s predecessor (V2V source)')
shutil.copy(f'{ROOT}/cache/sac/ckpt_137034.zip', f'{OUT}/models/sac_ckpt_137034.zip')
