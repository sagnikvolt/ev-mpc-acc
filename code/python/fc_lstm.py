"""BiLSTM forecaster, cross-fitted. Usage: python3 python/fc_lstm.py <fold>. Input sequence: v_{t-k} - v_t... reconstructed from features (21 -> 21 steps of [speed_rel, v_t]) ; hidden 64, 1 layer bi, MLP head -> 30 outputs. 8 epochs, Adam 2e-3, seed 42."""
import sys, os, time, numpy as np, pandas as pd, torch, torch.nn as nn
ROOT = os.path.expanduser('~/mnt/EV Project'); f = int(sys.argv[1]); torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(2)
T = np.load(f'{ROOT}/cache/fc_train.npz'); E = np.load(f'{ROOT}/cache/fc_eval.npz'); W = pd.read_csv(f'{ROOT}/results/windows.csv')
tr = np.where(T['fold'] != f)[0]; te = np.where(W.fold.values[E['win']] == f)[0]
def seq(X):   # X[:,0]=v_t, X[:,1+k-1]=v_t - v_{t-k}; sequence oldest->newest of v (relative to v_t) plus abs v_t channel
    rel = -X[:, 1:][:, ::-1]; rel = np.concatenate([rel, np.zeros((len(X), 1), np.float32)], 1)   # (n,21)  v_{t-k}-v_t, ending 0
    return np.stack([rel, np.repeat(X[:, :1], 21, 1)], 2).astype(np.float32)
Str, Ste = seq(T['X'][tr]), seq(E['X'][te]); sc = np.array([5.0, 20.0], np.float32); Str /= sc; Ste /= sc
Ytr = T['Y'][tr].astype(np.float32); ys = Ytr.std(0) + 1e-6; Ytr = Ytr / ys
class M(nn.Module):
    def __init__(s): super().__init__(); s.l = nn.LSTM(2, 64, batch_first=True, bidirectional=True); s.h = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 30))
    def forward(s, x): o, _ = s.l(x); return s.h(o[:, -1])
m = M(); opt = torch.optim.Adam(m.parameters(), 2e-3); t0 = time.time(); n = len(tr); bs = 512
Xt, Yt = torch.tensor(Str), torch.tensor(Ytr)
for ep in range(8):
    perm = torch.randperm(n); m.train()
    for b in range(0, n, bs):
        i = perm[b:b + bs]; opt.zero_grad(); loss = nn.functional.smooth_l1_loss(m(Xt[i]), Yt[i]); loss.backward(); opt.step()
    if time.time() - t0 > 140: print('time cap at epoch', ep + 1); break
m.eval()
with torch.no_grad(): P = m(torch.tensor(Ste)).numpy() * ys
np.save(f'{ROOT}/cache/fc_pred_bilstm_f{f}.npy', P.astype(np.float32)); print('bilstm fold', f, 'time %.0fs' % (time.time() - t0))
