"""Local test of the demo handler against the stored paper results (no AWS needed). Usage: python3 aws_demo/test_local.py [N=2]"""
import os, sys, json, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE); N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
sys.path.insert(0, f'{HERE}/app'); import handler as H
X = np.load(f'{PKG}/derived_data/openacc_lead_windows_203.npz')['lead_speed_mps'].astype(float)
W = pd.read_csv(f'{PKG}/derived_data/windows.csv'); test = [i for i in range(len(W)) if W.fold[i] > 0][:N]
ref = pd.read_csv(f'{PKG}/results/gate3b/r0_0.65_acc_0.csv').set_index('win')
refo = pd.concat([pd.read_csv(f) for f in glob.glob(f'{PKG}/results/gate3b/r0_0.65_oracle_s1_n27_*.csv')]).drop_duplicates('win').set_index('win')
ok = True
for w in test:
    r = H.lambda_handler({'body': json.dumps({'speeds': X[w].tolist(), 'hz': 10, 'eta': 0.65, 'cap_kw': 35, 'preview': 'oracle'})})
    b = json.loads(r['body']); good = r['statusCode'] == 200 and abs(b['acc']['whkm_adj'] - ref.loc[w, 'whkm_adj']) < 1e-3 and abs(b['mpc']['whkm_adj'] - refo.loc[w, 'whkm_adj']) < 0.05
    ok &= good; print(w, r['statusCode'], f"ACC {b['acc']['whkm_adj']:.3f}/{ref.loc[w,'whkm_adj']:.3f}  MPC {b['mpc']['whkm_adj']:.3f}/{refo.loc[w,'whkm_adj']:.3f}  change {b['energy_change_pct']:+.2f}%  trace pts {len(b['trace']['t'])}", 'OK' if good else 'MISMATCH')
# breaker option runs and is safe
bk = H.lambda_handler({'body': json.dumps({'speeds': X[test[0]].tolist(), 'preview': 'const', 'breakers': [200, 600]})}); b = json.loads(bk['body']); print('breakers', bk['statusCode'], {k: round(b[k]['whkm_adj'], 2) for k in ('acc', 'mpc')}, 'change', b['energy_change_pct'], 'ncross', b['mpc']['ncross'], 'lead min', min(b['trace']['lead'])); ok &= bk['statusCode'] == 200 and min(b['trace']['lead']) < 6 and b['mpc']['viol'] == 0
# input checks
for name, body, code in [('too short', {'speeds': [10] * 100}, 400), ('too long', {'speeds': [10] * 2000}, 400), ('too fast', {'speeds': [60] * 1200}, 400), ('bad eta', {'speeds': [10] * 1200, 'eta': 5}, 400), ('not json', 'zzz', 400)]:
    r = H.lambda_handler({'body': body if isinstance(body, str) else json.dumps(body)}); g = r['statusCode'] == code; ok &= g; print(name, r['statusCode'], 'OK' if g else 'WRONG')
print('PASS' if ok else 'FAIL')
