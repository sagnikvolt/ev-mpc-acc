"""LGBM V2V forecaster cross-fit. Usage: python3 python/v2v_lgbm.py <fold>  (train on folds != fold, predict eval rows of windows in that fold)"""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb
ROOT = os.path.expanduser('~/mnt/EV Project'); f = int(sys.argv[1]); T = np.load(f'{ROOT}/cache/v2v_train.npz'); E = np.load(f'{ROOT}/cache/v2v_eval.npz'); V = pd.read_csv(f'{ROOT}/results/v2v_windows.csv')
tr = np.where(T['fold'] != f)[0]; te = np.where(V.fold.values[E['win']] == f)[0]; J = [0, 1, 3, 5, 9, 13, 19, 29]; out = np.zeros((len(te), len(J)))
for a, j in enumerate(J):
    out[:, a] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1).fit(T['X'][tr].astype(np.float64), T['Y'][tr][:, j]).predict(E['X'][te].astype(np.float64))
np.save(f'{ROOT}/cache/v2v_pred_lgbm_f{f}.npy', np.stack([np.interp(np.arange(30), J, o) for o in out]).astype(np.float32)); np.save(f'{ROOT}/cache/v2v_rows_f{f}.npy', te); print('lgbm v2v fold', f, len(tr), len(te))
