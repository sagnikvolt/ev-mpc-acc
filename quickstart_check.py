"""Self-contained reproduction check using ONLY files in this package (no raw third-party data needed).
Reruns the conventional ACC and the oracle-preview MPC (eta 0.65, 35 kW cap) on N test windows and compares with the stored results.
Usage: python3 quickstart_check.py [N=3]      (about 5 s per MPC window)"""
import os, sys, tempfile, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
tmp = tempfile.mkdtemp(); os.makedirs(f'{tmp}/mnt/EV Project/results')          # the scripts read ~/mnt/EV Project/results/bmw_i3_params_final.npy
np.save(f'{tmp}/mnt/EV Project/results/bmw_i3_params_final.npy', pd.read_csv(f'{HERE}/derived_data/bmw_i3_params_final.csv').value.values)
os.environ['HOME'] = tmp; sys.path.insert(0, f'{HERE}/code/python')
import mpc_acc as A, baselines as B
A.set_regen(0.65)
X = np.load(f'{HERE}/derived_data/openacc_lead_windows_203.npz')['lead_speed_mps'].astype(float)
W = pd.read_csv(f'{HERE}/derived_data/windows.csv'); test = [i for i in range(len(W)) if W.fold[i] > 0][:N]
import glob
ref = pd.read_csv(f'{HERE}/results/gate3b/r0_0.65_acc_0.csv').set_index('win')
refo = pd.concat([pd.read_csv(f) for f in glob.glob(f'{HERE}/results/gate3b/r0_0.65_oracle_s1_n27_*.csv')]).drop_duplicates('win').set_index('win')
print('win  ACC Wh/km (rerun / stored); oracle checked against stored too   oracle vs ACC %')
ok = True
for w in test:
    a = B.simulate(X[w], B.acc_policy(0.4, 1.0)); o = A.simulate(X[w], A.MPC(), preview='oracle')
    d = (o['whkm_adj'] / a['whkm_adj'] - 1) * 100; same = abs(a['whkm_adj'] - ref.loc[w, 'whkm_adj']) < 1e-3 and abs(o['whkm_adj'] - refo.loc[w, 'whkm_adj']) < 0.05; ok &= same
    print(f'{w:3d}  {a["whkm_adj"]:.3f} / {ref.loc[w, "whkm_adj"]:.3f}   {d:+.2f}   {"OK" if same else "MISMATCH"}')
shutil.rmtree(tmp, ignore_errors=True); print('PASS' if ok else 'FAIL')
