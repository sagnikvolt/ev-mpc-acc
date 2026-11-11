"""Day-based folds (pre-registered 2026-11-08, Sonnet; before any tuning/forecaster results).
fold 0 = TUNE (MPC weights tuned once): 28_2_2019, 27_2_2019, 30_9_2020, 8_10_2019   (43 windows)
fold 1 = 24_9_2018 (Cherasco)  fold 2 = 26_2_2019 + 30_8_2018  fold 3 = 9_10_2019 (ZalaZone)  fold 4 = 27_10_2020 (Casale)
fold -1 = 4_7_2019, 5_7_2019 (AstaZero; no eval windows, always training-only)
Forecaster for test fold f is trained on all segments (any pair) whose fold != f (cross-fitted), so test days never leak.
Run: python3 python/folds.py -> results/windows.csv (203 eval windows: pair0 human-led, 120 s, mean>5 m/s, std>1 m/s), results/seg_folds.csv"""
import os, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/EV Project')
F = {'28_2_2019': 0, '27_2_2019': 0, '30_9_2020': 0, '8_10_2019': 0, '24_9_2018': 1, '26_2_2019': 2, '30_8_2018': 2, '9_10_2019': 3, '27_10_2020': 4, '4_7_2019': -1, '5_7_2019': -1}
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True)
pd.DataFrame({'seg': range(len(S)), 'day': [s['day'] for s in S], 'fold': [F[s['day']] for s in S]}).to_csv(f'{ROOT}/results/seg_folds.csv', index=False)
rows = []
for si, s in enumerate(S):
    if s['pair'] != 0 or s['lead_acc'] > 0.5: continue
    v = s['vl']
    for w0 in range(0, len(v) - 1200 + 1, 1200):
        x = v[w0:w0 + 1200]
        if x.mean() > 5 and x.std() > 1.0: rows.append((si, w0, s['campaign'], s['day'], F[s['day']]))
W = pd.DataFrame(rows, columns=['seg', 'w0', 'campaign', 'day', 'fold']); W.to_csv(f'{ROOT}/results/windows.csv', index=False)
print(len(W), 'windows;', W.fold.value_counts().sort_index().to_dict())
