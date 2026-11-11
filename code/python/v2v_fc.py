"""V2V forecasters. Channel: predecessor-of-lead speed received with 0.2 s delay (2 samples) and 10% packet loss (hold last value), per segment seed.
Features (43): lead 21 [v_t, v_t - v_{t-k*0.5s}] + pred 22 [vp_rx - v_t, vp_rx(t) - vp_rx(t-k*0.5 s) k=1..20]. Targets Y(30) as in fc_data.
Models: kinematic lag model (lead relaxes toward extrapolated predecessor speed; params fitted on folds 0,-1 only) and LGBM cross-fitted (test folds 1-3 + tune fold 0).
Out: cache/v2v_train.npz, v2v_eval.npz, v2v_kin.json, v2v_pred_lgbm_f{f}.npy ; prints RMSE table."""
import os, sys, json, time, numpy as np, pandas as pd
ROOT = os.path.expanduser('~/mnt/EV Project'); N = 30
def channel(vp, seed):
    r = np.random.default_rng(seed); d = np.concatenate([[vp[0]] * 2, vp[:-2]]); drop = r.random(len(vp)) < 0.10; rx = d.copy()
    for i in range(1, len(rx)):
        if drop[i]: rx[i] = rx[i - 1]
    return rx
def feats(vl, rx, i):
    idx = np.maximum(i - 5 * np.arange(1, 21), 0)
    return np.concatenate([[vl[i]], vl[i] - vl[idx], [rx[i] - vl[i]], rx[i] - rx[idx]])
def targ(v, i): return np.array([v[i + 5 * j: i + 5 * j + 6].mean() for j in range(N)]) - v[i]
def kin_pred(X, p):
    Tl, tau, wa = p; t = (np.arange(N) + 0.5) * 0.5; ap = np.clip(X[:, 23], -3.5, 2.0); al = np.clip(X[:, 2], -3.5, 2.0)
    lag = (X[:, 21] + tau * ap)[:, None] * (1 - np.exp(-t / Tl))[None, :]; ca = al[:, None] * 2.0 * (1 - np.exp(-t / 2.0))[None, :]
    return (1 - wa) * lag + wa * ca
if __name__ == '__main__':
    S = np.load(f'{ROOT}/cache/v2v_segs.npy', allow_pickle=True); V = pd.read_csv(f'{ROOT}/results/v2v_windows.csv'); RX = [channel(s['vp'], 1000 + si) for si, s in enumerate(S)]
    X, Y, fo = [], [], []
    for si, s in enumerate(S):
        v = s['vl']
        for i in range(100, len(v) - 5 * N - 1, 10): X.append(feats(v, RX[si], i)); Y.append(targ(v, i)); fo.append(s['fold'])
    np.savez_compressed(f'{ROOT}/cache/v2v_train.npz', X=np.array(X, np.float32), Y=np.array(Y, np.float32), fold=np.array(fo)); print('train', len(X))
    Xe, Ye, we, st = [], [], [], []
    for wi, r in V.iterrows():
        s = S[int(r.seg)]; nst = (1200 - N * 5 - 1) // 5
        for k in range(nst): i = int(r.w0) + 5 * k; Xe.append(feats(s['vl'], RX[int(r.seg)], i)); Ye.append(targ(s['vl'], i)); we.append(wi); st.append(k)
    np.savez_compressed(f'{ROOT}/cache/v2v_eval.npz', X=np.array(Xe, np.float32), Y=np.array(Ye, np.float32), win=np.array(we), step=np.array(st)); print('eval', len(Xe))
    X, Y, fo = np.array(X, np.float64), np.array(Y, np.float64), np.array(fo); tr = np.isin(fo, [0, -1])
    best = None
    for Tl in [1.5, 2, 3, 4, 6, 10]:
        for tau in [0, 1, 2, 3]:
            for wa in [0, 0.5, 1.0]:
                P = kin_pred(X[tr], (Tl, tau, wa)); e = np.mean([(P[:, j] - Y[tr][:, j]) ** 2 for j in (5, 9, 19)])
                if best is None or e < best[0]: best = (e, Tl, tau, wa)
    kp = best[1:]; json.dump(dict(Tl=kp[0], tau=kp[1], wa=kp[2]), open(f'{ROOT}/cache/v2v_kin.json', 'w')); print('kinematic V2V params (fit on folds 0,-1): Tl %.1f tau %.0f wa %.1f' % kp)
