"""Car-following env for SAC. Train segments: folds 0 and -1 ONLY (test days never seen). 0.5 s control, 60 s episodes, same plant/limits as baselines.simulate.
Action in [-1,1] -> delta accel = 1.5*act per step (jerk limit 3 m/s^3 built in); a = clip(a_prev+da, -3.5, 2).
Reward = -Wh(0.5 s)/2 - 0.02*min(excess,30)^2 (outside gap band 3+0.25 v around 5+1.5 v) - 0.3*min(shortfall,20) (m below 4+0.7 v) - 0.05*|da| ; collision (d<0.5) -> -300 - 3*remaining steps, terminate. (v2 after v1 learned to crash early)"""
import os, numpy as np, pandas as pd, gymnasium as gym
from gymnasium import spaces
import mpc_acc as A
ROOT = os.path.expanduser('~/mnt/EV Project'); TH, D0 = 1.5, 5.0
def obs_of(v, d, vl, al, a_prev): return np.array([v / 30, (vl - v) / 10, (d - (D0 + TH * v)) / 30, d / 60, vl / 30, al / 3, a_prev / 3], dtype=np.float32)
class Env(gym.Env):
    def __init__(s, folds=(0, -1), seed=0):
        S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); F = pd.read_csv(f'{ROOT}/results/seg_folds.csv').fold.values
        s.segs = [x['vl'] for x, f in zip(S, F) if f in folds and len(x['vl']) >= 700 and x['vl'].mean() > 2]
        s.rng = np.random.default_rng(seed); s.observation_space = spaces.Box(-5, 5, (7,), np.float32); s.action_space = spaces.Box(-1, 1, (1,), np.float32)
    def reset(s, seed=None, options=None):
        v = s.segs[s.rng.integers(len(s.segs))]; i0 = s.rng.integers(0, len(v) - 650); s.vl = v[i0:i0 + 650]; s.k = 0
        s.v = float(s.vl[0]); s.d = D0 + TH * s.v * s.rng.uniform(0.7, 1.4); s.a = 0.0; return s._obs(), {}
    def _obs(s):
        i = s.k * 5; al = np.clip((s.vl[i] - s.vl[max(i - 10, 0)]) / 1.0, -3.5, 2.0); return obs_of(s.v, s.d, s.vl[i], al, s.a)
    def step(s, act):
        da = 1.5 * float(np.clip(act[0], -1, 1)); a = float(np.clip(s.a + da, -3.5, 2.0))
        if s.v >= 38 and a > 0: a = 0.0
        i = s.k * 5; E = 0.0
        for q in range(5):
            v1 = max(0.0, s.v + a * 0.1); E += float(A.pb_kw(A.pw_kw(s.v, v1, 0.1), 0.5 * (s.v + v1))) * 0.1 / 3.6; s.d += (s.vl[i + q] - 0.5 * (s.v + v1)) * 0.1; s.v = v1
        band = 3 + 0.25 * s.v; e = s.d - (D0 + TH * s.v); exc = max(0.0, abs(e) - band); sh = max(0.0, 4 + 0.7 * s.v - s.d)
        r = -E / 2 - 0.02 * min(exc, 30.0) ** 2 - 0.3 * min(sh, 20.0) - 0.05 * abs(da); s.a = a; s.k += 1
        col = s.d < 0.5
        if col: r -= 300.0 + 3.0 * (120 - s.k)   # collision costs more than any amount of continued penalty
        return s._obs(), r, bool(col), bool(s.k >= 120), {}
def sac_policy(model):
    def pi(st):
        o = obs_of(st['v'], st['d'], st['vl'], st['al'], st['a_prev']); act, _ = model.predict(o, deterministic=True); return st['a_prev'] + 1.5 * float(act[0])
    return pi
