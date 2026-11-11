"""RMSE of lead-speed forecasts on TEST folds (1-4), overall and by dynamics tercile. Horizons: j=1 (1 s), 5 (3 s), 9 (5 s), 19 (10 s). Writes results/gate3b/fc_rmse.csv"""
import os, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/EV Project'); E = np.load(f'{ROOT}/cache/fc_eval.npz'); W = pd.read_csv(f'{ROOT}/results/windows.csv')
Y = E['Y'].astype(np.float64); X = E['X'].astype(np.float64); terc = E['terc'][E['win']]; test = W.fold.values[E['win']] > 0
t = (np.arange(30) + 0.5) * 0.5; a = np.clip(X[:, 2] / 1.0, -3.5, 2.0)    # X[:,2] = v_t - v_{t-1 s}
# pad-free: caccel mean-speed increment relative to v_t  (tau=2 s) -- same as closed-loop fc_ca
P = {'const-speed': np.zeros_like(Y), 'const-accel': a[:, None] * 2.0 * (1 - np.exp(-t / 2.0))[None, :]}
for m in ['ridge', 'knn', 'svr', 'lgbm', 'bilstm']:
    A = np.full_like(Y, np.nan)
    for f in [1, 2, 3, 4]:
        rows = np.load(f'{ROOT}/cache/fc_rows_f{f}.npy') if m != 'bilstm' else np.where(W.fold.values[E['win']] == f)[0]; A[rows] = np.load(f'{ROOT}/cache/fc_pred_{m}_f{f}.npy')
    P[m] = A
J = {'1 s': 1, '3 s': 5, '5 s': 9, '10 s': 19}; rows = []
for grp, mask in [('all', test), ('calm', test & (terc == 0)), ('medium', test & (terc == 1)), ('dynamic', test & (terc == 2))]:
    for mname, A in P.items():
        r = dict(group=grp, model=mname, n=int(mask.sum()))
        for h, j in J.items(): r[h] = float(np.sqrt(np.mean((A[mask, j] - Y[mask, j]) ** 2)))
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f'{ROOT}/results/gate3b/fc_rmse.csv', index=False)
for grp in ['all', 'calm', 'medium', 'dynamic']:
    g = R[R.group == grp].set_index('model')[list(J)]; kin = g.loc[['const-speed', 'const-accel']].min()
    print(f'--- {grp} (n={R[R.group==grp].n.iloc[0]}) RMSE m/s; last block = % better than BEST kinematic'); print(g.round(3).to_string()); print((100 * (1 - g / kin)).round(1).iloc[2:].to_string())
