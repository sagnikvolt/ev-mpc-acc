"""V2V data: for every platoon file and every vehicle k>=1 (0-based) take lead = vehicle k, predecessor-of-lead = vehicle k-1 (both 10 Hz, finite, contiguous >=30 s).
The simulated ego car follows `lead`; the recorded follower is not used. Folds by recording day as in folds.py. Windows: 120 s non-overlapping, mean>5 m/s, std>1 m/s.
Out: cache/v2v_segs.npy, results/v2v_windows.csv (role=test: 30 random per test fold 1,2,3 (seed 42); role=tune: 30 random from fold 0 (seed 42)); written BEFORE any V2V run."""
import os, re, glob, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/EV Project'); RAW = f'{ROOT}/data/raw/openacc'; SKIP = {'One_vehicle_multiple_drivers_on_road_campaign'}
FOLD = {'28_2_2019': 0, '27_2_2019': 0, '30_9_2020': 0, '8_10_2019': 0, '24_9_2018': 1, '26_2_2019': 2, '30_8_2018': 2, '9_10_2019': 3, '27_10_2020': 4, '4_7_2019': -1, '5_7_2019': -1}
def read(path):
    lines = open(path, errors='ignore').read().splitlines(); hi = max(i for i, l in enumerate(lines[:12]) if l.startswith('Time,')); meta = {}
    for l in lines[:hi]:
        p = l.split(',')
        if p[0] in ('Date', 'Vehicle_order'): meta[p[0]] = [x for x in p[1:] if x != '']
    return meta, pd.read_csv(path, skiprows=hi, low_memory=False)
segs = []
for camp in sorted(os.listdir(RAW)):
    if not os.path.isdir(f'{RAW}/{camp}') or camp in SKIP: continue
    for path in sorted(glob.glob(f'{RAW}/{camp}/*.csv')):
        name = os.path.basename(path)
        if 'spec' in name.lower(): continue
        try: meta, df = read(path)
        except Exception as e: print('ERR', name, e); continue
        sp = sorted([c for c in df.columns if re.fullmatch(r'Speed\d+', c)], key=lambda c: int(c[5:]))
        if len(sp) < 2: continue
        day = '_'.join(meta.get('Date', ['?'])); t = df['Time'].values.astype(float)
        for k in range(1, len(sp)):
            vp, vl = df[sp[k - 1]].values.astype(float), df[sp[k]].values.astype(float)
            dr = df[f'Driver{k+1}'].astype(str).values if f'Driver{k+1}' in df.columns else np.array(['?'] * len(t))
            idx = np.where(np.isfinite(vp) & np.isfinite(vl) & np.isfinite(t))[0]
            if len(idx) < 300: continue
            for b in np.split(idx, np.where((np.diff(t[idx]) > 0.15) | (np.diff(idx) > 1))[0] + 1):
                if len(b) < 300: continue
                segs.append(dict(campaign=camp, file=name, day=day, fold=FOLD[day], k=k, vl=vl[b], vp=vp[b], lead_acc=float(np.mean(np.array([str(x).upper() for x in dr[b]]) == 'ACC'))))
np.save(f'{ROOT}/cache/v2v_segs.npy', np.array(segs, dtype=object), allow_pickle=True)
rows = []
for si, s in enumerate(segs):
    for w0 in range(0, len(s['vl']) - 1200 + 1, 1200):
        x = s['vl'][w0:w0 + 1200]
        if x.mean() > 5 and x.std() > 1.0: rows.append(dict(seg=si, w0=w0, fold=s['fold'], campaign=s['campaign'], day=s['day'], k=s['k'], lead_acc=s['lead_acc']))
A = pd.DataFrame(rows); print(len(segs), 'segments;', len(A), 'candidate windows'); print(A.groupby(['fold', 'campaign']).size())
rng = np.random.default_rng(42); sel = []
for f in [1, 2, 3]: sel.append(A[A.fold == f].sample(min(30, (A.fold == f).sum()), random_state=42).assign(role='test'))
sel.append(A[A.fold == 0].sample(30, random_state=42).assign(role='tune'))
V = pd.concat(sel).reset_index(drop=True); V.to_csv(f'{ROOT}/results/v2v_windows.csv', index=False); A.to_csv(f'{ROOT}/results/v2v_candidates.csv', index=False)
print(V.groupby(['role', 'fold']).agg(n=('seg', 'size'), lead_acc=('lead_acc', 'mean')))
