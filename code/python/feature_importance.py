"""Step C: which input features govern the battery-power prediction (permutation importance) + learning curve + dataset sizes.
Same data, folds, models and settings as python/ml_bench.py (BMW i3, 68 trips, 2 Hz, 5-fold trip-wise CV, seed 42). Importance = increase in power RMSE (kW)
on HELD-OUT trips when one feature GROUP is shuffled (grouped permutation, because speed/accel features are lagged copies of each other).
Usage: python3 python/feature_importance.py imp <lgbm_pure|lgbm_resid|svr_resid|knn_resid> | curve | sizes"""
import sys, os, json, time, numpy as np, pandas as pd
os.chdir(os.path.expanduser('~/mnt/EV Project'))
d = np.load('cache/ml_data.npz', allow_pickle=True); X = d['X'].astype(np.float64); Pb = d['Pb']; trip = d['trip']; ft = d['fold_of_trip']; fold = ft[trip]; fn = [str(x) for x in d['feat_names']]
GROUPS = {'Speed (v, v2, lags)': ['v', 'v2', 'v_l2', 'v_l5', 'v_m5'], 'Acceleration (a, av, lags)': ['a', 'av', 'a_l2', 'a_l5', 'a_m5'], 'Road grade': ['gr'], 'Cabin heating': ['heat'], 'Air-conditioning': ['ac'],
          'Ambient temperature': ['amb'], 'Battery temperature': ['bt'], 'State of charge': ['soc']}
def prep(tr, te):
    med = np.nanmedian(X[tr], axis=0); med = np.where(np.isnan(med), 0, med); f = lambda A: np.where(np.isnan(A), med, A); mu = f(X[tr]).mean(0); sd = f(X[tr]).std(0) + 1e-9
    return (f(X[tr]) - mu) / sd, (f(X[te]) - mu) / sd
def build(k, resid):
    tr = np.where(fold != k)[0]; te = np.where(fold == k)[0]; Pp = d['Pphys_%d' % k]; Xtr, Xte = prep(tr, te); cols = list(fn)
    if resid:
        Pw = d['Pw_%d' % k]; m_, s_ = Pw[tr].mean(), Pw[tr].std() + 1e-9; Xtr = np.column_stack([Xtr, (Pw[tr] - m_) / s_]); Xte = np.column_stack([Xte, (Pw[te] - m_) / s_]); cols = cols + ['Pw']
    target = (Pb - Pp) if resid else Pb; return tr, te, Xtr, Xte, target, cols
def fit(model, Xtr, ytr, rng):
    if model == 'lgbm':
        import lightgbm as lgb; return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=2, verbose=-1).fit(Xtr, ytr), lambda m, A: m.predict(A)
    if model == 'knn':
        from sklearn.neighbors import KNeighborsRegressor; sel = rng.choice(len(Xtr), min(70000, len(Xtr)), replace=False); m = KNeighborsRegressor(n_neighbors=25, weights='distance', n_jobs=2).fit(Xtr[sel], ytr[sel]); return m, lambda m, A: m.predict(A)
    from sklearn.svm import SVR; sel = rng.choice(len(Xtr), 8000, replace=False); sc = np.std(ytr[sel]); m = SVR(kernel='rbf', C=3.0, epsilon=0.05, gamma='scale').fit(Xtr[sel], ytr[sel] / sc); return m, lambda m, A: m.predict(A) * sc
if sys.argv[1] == 'imp':
    name = sys.argv[2]; model, mode = name.split('_'); resid = mode == 'resid'; rng = np.random.default_rng(42); prng = np.random.default_rng(7)
    nrep = 3 if model == 'lgbm' else 2; nsub = None if model == 'lgbm' else 5000; rows = []; FO = [int(x) for x in os.environ.get('FOLDS', '0,1,2,3,4').split(',')]; t0 = time.time()
    for k in FO:
        tr, te, Xtr, Xte, target, cols = build(k, resid); m, pr = fit(model, Xtr, target[tr], rng)
        idx = np.arange(len(te)) if nsub is None else np.sort(prng.choice(len(te), min(nsub, len(te)), replace=False)); A = Xte[idx]; y = target[te][idx]
        base = float(np.sqrt(np.mean((pr(m, A) - y) ** 2)) / 1000); groups = dict(GROUPS)
        if resid: groups['Physics wheel power (Pw)'] = ['Pw']
        for g, fs in groups.items():
            ci = [cols.index(f) for f in fs]; inc = []
            for r in range(nrep):
                B = A.copy(); p = prng.permutation(len(B)); B[:, ci] = A[p][:, ci]; inc.append(float(np.sqrt(np.mean((pr(m, B) - y) ** 2)) / 1000) - base)
            rows.append(dict(model=name, fold=k, group=g, base_rmse=base, increase_kw=float(np.mean(inc))))
        print(name, 'fold', k, 'base %.2f kW' % base, '%.0fs' % (time.time() - t0), flush=True)
    pd.DataFrame(rows).to_csv('results/ml/importance_%s%s.csv' % (name, '' if len(FO) == 5 else '_f' + ''.join(map(str, FO))), index=False)
elif sys.argv[1] == 'curve':
    import lightgbm as lgb; rows = []; sizes = [10000, 25000, 50000, 100000, 10 ** 9]
    FO = [int(x) for x in os.environ.get('FOLDS', '0,1,2,3,4').split(',')]; MODES = os.environ.get('MODE', 'resid,pure').split(',')
    for resid in [m == 'resid' for m in MODES]:
        for k in FO:
            tr, te, Xtr, Xte, target, cols = build(k, resid); trips_tr = np.unique(trip[tr]); y_te = target[te]
            for rep in range(3):
                order = np.random.default_rng(100 + rep + 10 * k).permutation(trips_tr)
                for n in sizes:
                    chosen = []; cnt = 0
                    for t in order:
                        chosen.append(t); cnt += int((trip[tr] == t).sum())
                        if cnt >= n: break
                    msk = np.isin(trip[tr], chosen); m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, n_jobs=2, verbose=-1).fit(Xtr[msk], target[tr][msk])
                    rows.append(dict(mode='resid' if resid else 'pure', fold=k, rep=rep, n_target=n if n < 10 ** 9 else 0, n_train=int(msk.sum()), n_trips=len(chosen), rmse_kw=float(np.sqrt(np.mean((m.predict(Xte) - y_te) ** 2)) / 1000)))
            print('curve', 'resid' if resid else 'pure', 'fold', k, flush=True)
    pd.DataFrame(rows).to_csv('results/ml/learning_curve_%s_f%s.csv' % ('_'.join(MODES), ''.join(map(str, FO))), index=False)
else:
    z = pd.read_csv('data/processed/zoe_trip_summary.csv'); zr = len(pd.read_csv('data/processed/zoe_trips.csv', usecols=[0])); o = pd.read_csv('results/openacc_summary.csv'); W = pd.read_csv('results/windows.csv'); V = pd.read_csv('results/v2v_windows.csv')
    zt = pd.read_csv('data/processed/zoe_trips.csv', usecols=['trip_id', 'time_in_trip_s']); dt = float(zt.groupby('trip_id').time_in_trip_s.apply(lambda s: s.diff().median()).median())
    rows = [dict(dataset='BMW i3 (TU Munich, measured battery power) - energy-model data', unit='samples at 2 Hz', samples=len(Pb), trips_or_segments=int(len(np.unique(trip))), hours=round(len(Pb) / 2 / 3600, 1), use='physics fit, residual ML, feature importance, learning curve'),
            dict(dataset='Renault Zoe (public trip data)', unit='samples at 1 Hz', samples=zr, trips_or_segments=int(len(z)), hours=round(float(z.duration_s.sum()) / 3600, 1), use='second-vehicle dataset (cleaned, trip-wise split); not used in the tables here'),
            dict(dataset='OpenACC (JRC real car-following, 10 Hz)', unit='samples at 10 Hz', samples=int(o.n.sum()), trips_or_segments=int(len(o)), hours=round(float(o.dur_s.sum()) / 3600, 1), use='lead-vehicle forecasting, controller tests'),
            dict(dataset='OpenACC 120 s windows, folds 0-4 (all)', unit='windows', samples=len(W), trips_or_segments=int(W.seg.nunique()), hours=round(len(W) * 120 / 3600, 1), use='closed-loop experiments (43 tune, 160 test)'),
            dict(dataset='V2V 120 s windows (30 tune, 90 test) from 961 segments', unit='windows', samples=len(V), trips_or_segments=961, hours=round(len(V) * 120 / 3600, 1), use='V2V forecasting and closed-loop check')]
    D = pd.DataFrame(rows); D.to_csv('results/ml/dataset_sizes.csv', index=False); print(D.to_string(index=False))
