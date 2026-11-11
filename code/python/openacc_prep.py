"""Extract lead-vehicle speed series (10 Hz) from OpenACC platoon campaigns.
Output: cache/openacc_leads.npz  (object arrays of per-segment series) + results/openacc_summary.csv
Each segment = one (lead, follower) pair in one file, contiguous at 10 Hz.
Split key: campaign/day (for held-out evaluation)."""
import os, re, glob, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/EV Project')
RAW = f'{ROOT}/data/raw/openacc'
SKIP = {'One_vehicle_multiple_drivers_on_road_campaign'}  # single vehicle, no lead

def read(path):
    with open(path, errors='ignore') as f:
        lines = f.read().splitlines()
    hi = max(i for i, l in enumerate(lines[:12]) if l.startswith('Time,'))
    # header may appear twice (line 1 and in metadata end); use last within first 12 lines
    meta = {}
    for l in lines[:hi]:
        p = l.split(',')
        if p[0] in ('Date', 'Vehicle_order', 'Number_of_vehicles', 'ACC', 'Distance_setting'):
            meta[p[0]] = [x for x in p[1:] if x != '']
    df = pd.read_csv(path, skiprows=hi, low_memory=False)
    return meta, df

rows, segs = [], []
for camp in sorted(os.listdir(RAW)):
    d = f'{RAW}/{camp}'
    if not os.path.isdir(d) or camp in SKIP: continue
    for path in sorted(glob.glob(f'{d}/*.csv')):
        name = os.path.basename(path)
        if 'spec' in name.lower(): continue
        try: meta, df = read(path)
        except Exception as e:
            print('ERR', path, e); continue
        sp = sorted([c for c in df.columns if re.fullmatch(r'Speed\d+', c)], key=lambda c: int(c[5:]))
        if len(sp) < 2: continue
        order = meta.get('Vehicle_order', [])
        day = '_'.join(meta.get('Date', ['?']))
        t = df['Time'].values.astype(float)
        for k in range(len(sp) - 1):
            lead = df[sp[k]].values.astype(float); fol = df[sp[k+1]].values.astype(float)
            ivs = df[f'IVS{k+1}'].values.astype(float) if f'IVS{k+1}' in df.columns else np.full(len(t), np.nan)
            dr = df[f'Driver{k+2}'].astype(str).values.astype(object) if f'Driver{k+2}' in df.columns else np.array(['?']*len(t))
            ld = df[f'Driver{k+1}'].astype(str).values.astype(object) if f'Driver{k+1}' in df.columns else np.array(['?']*len(t))
            ok = np.isfinite(lead) & np.isfinite(fol) & np.isfinite(t)
            # contiguous blocks at 10 Hz
            idx = np.where(ok)[0]
            if len(idx) < 300: continue
            br = np.where((np.diff(t[idx]) > 0.15) | (np.diff(idx) > 1))[0] + 1
            for b in np.split(idx, br):
                if len(b) < 300: continue   # >= 30 s
                segs.append(dict(campaign=camp, file=name, day=day, pair=k, t=t[b], vl=lead[b], vf=fol[b],
                                 ivs=ivs[b], fol_acc=float(np.mean(np.array([str(x).upper() for x in dr[b]]) == 'ACC')),
                                 lead_acc=float(np.mean(np.array([str(x).upper() for x in ld[b]]) == 'ACC')),
                                 lead_veh=order[k] if k < len(order) else '?', fol_veh=order[k+1] if k+1 < len(order) else '?'))
                rows.append(dict(i=len(segs)-1, campaign=camp, file=name, day=day, pair=k, n=len(b), dur_s=len(b)/10,
                                 vmean=float(np.mean(lead[b])), vmax=float(np.max(lead[b])), fol_acc=segs[-1]['fol_acc'],
                                 lead_acc=segs[-1]['lead_acc'], lead_veh=segs[-1]['lead_veh'], fol_veh=segs[-1]['fol_veh']))
S = pd.DataFrame(rows)
os.makedirs(f'{ROOT}/cache', exist_ok=True)
np.save(f'{ROOT}/cache/openacc_segs.npy', np.array(segs, dtype=object), allow_pickle=True)
S.to_csv(f'{ROOT}/results/openacc_summary.csv', index=False)
print(len(S), 'segments;', S.n.sum(), 'samples;', round(S.dur_s.sum()/3600, 2), 'h')
print(S.groupby('campaign').agg(seg=('n', 'size'), hours=('dur_s', lambda x: x.sum()/3600), vmean=('vmean', 'mean'), vmax=('vmax', 'max')).round(2))
print(S.groupby('day').size().to_string())
