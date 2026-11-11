"""Step A forecasters, cross-fitted by day fold. Usage: python3 python/fc_bench.py <model> <fold>   (fold in 1..4; trained on folds != fold, incl. fold 0 and -1)
Models: ridge, knn, svr, lgbm. Saves cache/fc_pred_<model>_f<fold>.npy (rows of eval set in that fold)."""
import sys, os, time, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/EV Project'); model, f = sys.argv[1], int(sys.argv[2])
T = np.load(f'{ROOT}/cache/fc_train.npz'); E = np.load(f'{ROOT}/cache/fc_eval.npz'); W = pd.read_csv(f'{ROOT}/results/windows.csv')
tr = np.where(T['fold'] != f)[0]; te = np.where(W.fold.values[E['win']] == f)[0]
Xtr, Ytr, Xte = T['X'][tr].astype(np.float64), T['Y'][tr].astype(np.float64), E['X'][te].astype(np.float64)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
rng = np.random.default_rng(42); t0 = time.time(); J = [0, 1, 3, 5, 9, 13, 19, 29]
if model == 'ridge':
    from sklearn.linear_model import Ridge; P = Ridge(alpha=10.0).fit(Ztr, Ytr).predict(Zte)
elif model == 'knn':
    from sklearn.neighbors import KNeighborsRegressor; s = rng.choice(len(tr), min(100000, len(tr)), replace=False); P = KNeighborsRegressor(n_neighbors=25, weights='distance', n_jobs=2).fit(Ztr[s], Ytr[s]).predict(Zte)
elif model in ('svr', 'lgbm'):
    out = np.zeros((len(te), len(J)))
    if model == 'svr':
        from sklearn.svm import SVR; s = rng.choice(len(tr), 8000, replace=False)
        for a, j in enumerate(J): out[:, a] = SVR(kernel='rbf', C=3.0, epsilon=0.05, gamma='scale').fit(Ztr[s], Ytr[s, j]).predict(Zte)
    else:
        import lightgbm as lgb
        for a, j in enumerate(J):
            out[:, a] = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=2, verbose=-1).fit(Xtr, Ytr[:, j]).predict(Xte)
    P = np.stack([np.interp(np.arange(30), J, o) for o in out])
np.save(f'{ROOT}/cache/fc_pred_{model}_f{f}.npy', P.astype(np.float32)); np.save(f'{ROOT}/cache/fc_rows_f{f}.npy', te)
print(model, 'fold', f, 'train', len(tr), 'test', len(te), 'time %.0fs' % (time.time() - t0))
