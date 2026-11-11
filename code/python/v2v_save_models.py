"""Save the cross-fit LGBM V2V boosters (same hyper-params/seed as v2v_lgbm.py) so scenario runs can forecast online. Usage: python3 python/v2v_save_models.py <fold>"""
import sys, os, numpy as np, lightgbm as lgb, joblib
ROOT = os.path.expanduser('~/mnt/EV Project'); f = int(sys.argv[1]); T = np.load(f'{ROOT}/cache/v2v_train.npz'); tr = np.where(T['fold'] != f)[0]; J = [0, 1, 3, 5, 9, 13, 19, 29]
M = [lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1).fit(T['X'][tr].astype(np.float64), T['Y'][tr][:, j]) for j in J]
joblib.dump(dict(J=J, M=M), f'{ROOT}/cache/v2v_models_f{f}.pkl'); print('saved fold', f)
