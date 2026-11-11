"""SAC training in time-capped chunks (checkpoint between calls). Usage: python3 python/sac_train.py <seconds>
Checkpoint: cache/sac/model.zip + replay.pkl. After each chunk prints fold-0 validation (43 windows) so checkpoint choice never uses test windows."""
import sys, os, time, warnings, numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import sac_env as SE, baselines as B
torch.set_num_threads(2); ROOT = SE.ROOT; D = f'{ROOT}/cache/sac'; os.makedirs(D, exist_ok=True); cap = float(sys.argv[1])
env = SE.Env()
if os.path.exists(f'{D}/model.zip'):
    m = SAC.load(f'{D}/model.zip', env=env, device='cpu'); m.load_replay_buffer(f'{D}/replay.pkl')
else:
    m = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=200000, batch_size=256, learning_starts=2000, gamma=0.98, tau=0.01, policy_kwargs=dict(net_arch=[128, 128]), seed=42, device='cpu', verbose=0)
class Cap(BaseCallback):
    def __init__(s): super().__init__(); s.t0 = time.time()
    def _on_step(s): return time.time() - s.t0 < cap
m.learn(total_timesteps=10 ** 7, callback=Cap(), reset_num_timesteps=False, log_interval=10 ** 9)
m.save(f'{D}/model.zip'); m.save_replay_buffer(f'{D}/replay.pkl'); m.save(f'{D}/ckpt_{m.num_timesteps}.zip')
S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv'); pi = SE.sac_policy(m); R = []
for w in [w for w in range(len(W)) if W.fold[w] == 0]:
    r = W.iloc[w]; R.append(B.simulate(S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200], pi))
R = pd.DataFrame(R); print('timesteps', m.num_timesteps, '| fold0: whkm_adj %.2f short %.2f jerk %.2f dmin %.1f score %.2f (ACC ref 137.49/0.00/0.25)' % (R.whkm_adj.mean(), R.short_max.mean(), R.jerk_rms.mean(), R.dmin.mean(), R.whkm_adj.mean() + 2 * R.short_max.mean()))
