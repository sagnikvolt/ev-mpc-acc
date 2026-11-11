"""Forecasting dataset. Input: lead speed history (10 s at 0.5 s steps). Output: lead speed mean over each of next 30 intervals of 0.5 s, minus current speed.
Features (21): v_t, v_t - v_{t-k*0.5s} k=1..20. Targets Y (30): mean(vl[i+5j : i+5j+6]) - v_t, j=0..29 (identical to oracle preview in mpc).
Train set: every 1 s from every segment (any pair), tagged with day-fold. Eval set: every 0.5 s inside the 203 evaluation windows (closed-loop timing), tagged with window id.
Out: cache/fc_train.npz, cache/fc_eval.npz"""
import os, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/EV Project'); H = 20; N = 30
def feats(v, i):
    idx = i - 5 * np.arange(1, H + 1); idx = np.maximum(idx, 0)   # pad with first sample if history short
    return np.concatenate([[v[i]], v[i] - v[idx]])
def targ(v, i): return np.array([v[i + 5 * j: i + 5 * j + 6].mean() for j in range(N)]) - v[i]
if __name__ == '__main__':
    S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); SF = pd.read_csv(f'{ROOT}/results/seg_folds.csv'); W = pd.read_csv(f'{ROOT}/results/windows.csv')
    X, Y, fo, sg = [], [], [], []
    for si, s in enumerate(S):
        v = s['vl']
        for i in range(100, len(v) - 5 * N - 1, 10):
            X.append(feats(v, i)); Y.append(targ(v, i)); fo.append(SF.fold[si]); sg.append(si)
    np.savez_compressed(f'{ROOT}/cache/fc_train.npz', X=np.array(X, dtype=np.float32), Y=np.array(Y, dtype=np.float32), fold=np.array(fo), seg=np.array(sg)); print('train', len(X))
    X, Y, wi, ii = [], [], [], []; wst = []
    for w, r in enumerate(W.itertuples()):
        v = S[r.seg]['vl']; nst = (1200 - N * 5 - 1) // 5
        x = v[r.w0:r.w0 + 1200]; wst.append(np.sqrt(np.mean(np.diff(x[::5]) ** 2)) / 0.5)
        for k in range(nst):
            i = r.w0 + 5 * k; X.append(feats(v, i)); Y.append(targ(v, i)); wi.append(w); ii.append(k)
    wst = np.array(wst); T = W.fold.values > 0; q = np.quantile(wst[T], [1/3, 2/3]); terc = np.digitize(wst, q)   # 0 calm 1 medium 2 dynamic
    np.savez_compressed(f'{ROOT}/cache/fc_eval.npz', X=np.array(X, dtype=np.float32), Y=np.array(Y, dtype=np.float32), win=np.array(wi), step=np.array(ii), terc=terc, winacc=wst, q=q)
    print('eval', len(X), 'tercile cut (accel RMS)', q.round(3))
