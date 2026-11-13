"""Step C tables -> results/ml/feature_importance_summary.md  (run from project root)"""
import glob, pandas as pd, numpy as np
R = 'results/ml/'; out = []; P = out.append
imp = {}
for name in ('lgbm_pure', 'lgbm_resid', 'svr_resid', 'knn_resid'):
    imp[name] = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{R}importance_{name}*.csv'))]); assert imp[name].fold.nunique() == 5, name
P('# Which inputs govern the battery power (BMW i3, 68 trips, 5-fold trip-wise CV)\n')
P('Permutation importance on HELD-OUT trips: increase in power RMSE (kW) when one input group is shuffled; mean over 5 folds (3 repeats for LightGBM, 2 repeats on 5,000 test rows for SVR/KNN). Bigger = the model relies on it more. "Physics wheel power (Pw)" exists only in the residual models.\n')
groups = list(dict.fromkeys(g for n in imp for g in imp[n].group)); T = pd.DataFrame({n: imp[n].groupby('group').increase_kw.mean() for n in imp}).reindex(groups)
T = T.sort_values('lgbm_pure', ascending=False); P('| Input group | LightGBM pure (no physics) | LightGBM + physics | SVR + physics | KNN + physics |'); P('|---|---|---|---|---|')
for g, r in T.iterrows(): P('| %s | %s |' % (g, ' | '.join('-' if np.isnan(v) else '%.3f' % v for v in r)))
P('\nBase RMSE (kW, mean over folds): ' + ', '.join('%s %.2f' % (n, imp[n].groupby('fold').base_rmse.first().mean()) for n in imp))
pure = imp['lgbm_pure']; fo = pure.pivot(index='group', columns='fold', values='increase_kw'); P('\nLightGBM pure, per-fold ranking stability (rank of each group in each fold, 1 = most important):')
rk = fo.rank(ascending=False).astype(int); P(rk.loc[[i for i in T.index if i in rk.index]].to_string())
lc = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}learning_curve_*.csv')]); lc['size'] = lc.n_target.replace(0, np.nan)
g = lc.groupby(['mode', 'n_target']).agg(n_train=('n_train', 'mean'), trips=('n_trips', 'mean'), rmse=('rmse_kw', 'mean'), sd=('rmse_kw', 'std')).reset_index()
P('\n# Learning curve: how much data is enough (LightGBM, trip-wise subsets of the training trips, 5 folds x 3 draws)\n'); P('| Training samples (2 Hz) | Training trips | RMSE kW, physics + LightGBM | RMSE kW, pure LightGBM |'); P('|---|---|---|---|')
for n in sorted(g.n_target.unique(), key=lambda x: (x == 0, x)):
    a = g[(g['mode'] == 'resid') & (g.n_target == n)].iloc[0]; b = g[(g['mode'] == 'pure') & (g.n_target == n)].iloc[0]; P('| %s (%d) | %.0f | %.2f ± %.2f | %.2f ± %.2f |' % ('requested ' + ('full' if n == 0 else '%d' % n), round(a.n_train), a.trips, a.rmse, a.sd, b.rmse, b.sd))
P('\n# Dataset sizes\n'); D = pd.read_csv(R + 'dataset_sizes.csv'); P('| Dataset | Unit | Count | Trips / segments | Hours | Use |'); P('|---|---|---|---|---|---|')
for _, r in D.iterrows(): P('| %s | %s | %s | %s | %s | %s |' % (r.dataset, r.unit, format(int(r.samples), ','), r.trips_or_segments, r.hours, r.use))
open(R + 'feature_importance_summary.md', 'w').write('\n'.join(out)); print('\n'.join(out))
