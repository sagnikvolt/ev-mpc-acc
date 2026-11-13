"""Step B2: SAC with the control-barrier safety filter inside the environment, eta 0.65. See preregistrations/safe_rl_prereg.md.
Three seeds (42 primary, 43, 44) train in parallel (1 CPU thread each). Crash-safe: every 20k-step checkpoint + replay buffer is saved; re-running resumes.
Usage: python rerun_sac_eta065.py     (env SEEDS="42,43,44" to change the seed list, CHUNKS (default 10) x CHUNK_STEPS (default 20000))"""
import sys, os, time, json, warnings
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_k] = "1"
import numpy as np, pandas as pd
warnings.filterwarnings('ignore'); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.expanduser('~/mnt/EV Project'); OUT = f'{ROOT}/results/safe_rl'; CK = f'{ROOT}/cache/safe_rl'
GAMMA = json.load(open(f'{ROOT}/results/safe_rl/gamma.json'))['gamma']
if os.environ.get('SMOKE'): OUT += '_smoke'; CK += '_smoke'
CHUNK = int(os.environ.get('CHUNK_STEPS', '20000')); NCH = int(os.environ.get('CHUNKS', '10')); SEEDS = [int(x) for x in os.environ.get('SEEDS', '42,43,44').split(',')]
def fmt(s):
    s = int(s); return f'{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}' if s >= 3600 else f'{s // 60:02d}:{s % 60:02d}'
def log(msg): open(f'{ROOT}/rerun_log_safe_rl.txt', 'a').write(time.strftime('%H:%M:%S ') + msg + '\n')

def worker(seed):
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import SAC
    import mpc_acc as A, baselines as B, sac_env as SE, safe_filter as SF
    A.set_regen(0.65)
    D = f'{CK}/seed{seed}'; os.makedirs(D, exist_ok=True); os.makedirs(OUT, exist_ok=True); pf = f'{D}/progress.json'
    S = np.load(f'{ROOT}/cache/openacc_segs.npy', allow_pickle=True); W = pd.read_csv(f'{ROOT}/results/windows.csv')
    F0 = [w for w in range(len(W)) if W.fold[w] == 0]; TEST = [w for w in range(len(W)) if W.fold[w] > 0]
    def prog(**k):
        st = json.load(open(pf)) if os.path.exists(pf) else {}; st.update(k); json.dump(st, open(pf + '.tmp', 'w')); os.replace(pf + '.tmp', pf)
    def evaluate(m, wins):
        R = []
        for w in wins:
            r = W.iloc[w]; pi = SF.filtered_policy(m, GAMMA); o = B.simulate(S[int(r.seg)]['vl'][int(r.w0):int(r.w0) + 1200], pi); R.append(dict(win=w, **o, iv_rate=pi.iv / pi.n, infeasible=pi.inf))
        return pd.DataFrame(R)
    f0csv = f'{D}/fold0_scores.csv'; done = pd.read_csv(f0csv) if os.path.exists(f0csv) else pd.DataFrame()
    nd = len(done); env = SF.FEnv(gamma=GAMMA, seed=seed)
    if os.path.exists(f'{D}/model.zip') and nd > 0:
        m = SAC.load(f'{D}/model.zip', env=env, device='cpu'); m.load_replay_buffer(f'{D}/replay.pkl')
    else:
        nd = 0; done = pd.DataFrame()
        m = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=200000, batch_size=256, learning_starts=2000, gamma=0.98, tau=0.01,
                policy_kwargs=dict(net_arch=[128, 128]), seed=seed, device='cpu', verbose=0)
    prog(phase='train', steps=m.num_timesteps, chunks_done=nd, t_start=time.time())
    for c in range(nd + 1, NCH + 1):
        t0 = time.time()
        m.learn(total_timesteps=CHUNK, reset_num_timesteps=False, log_interval=10 ** 9, progress_bar=False)
        R = evaluate(m, F0); sc = float(R.whkm_adj.mean() + 2 * R.short_max.mean())
        row = dict(chunk=c, steps=int(m.num_timesteps), whkm=float(R.whkm_adj.mean()), short=float(R.short_max.mean()), jerk=float(R.jerk_rms.mean()), gap=float(R.gap_mean.mean()),
                   unsafe=int((R.short_max > 0.5).sum()), iv=float(R.iv_rate.mean()), score=sc, secs=time.time() - t0)
        m.save(f'{D}/ckpt_{m.num_timesteps}.zip'); m.save(f'{D}/model.zip'); m.save_replay_buffer(f'{D}/replay.pkl')
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(f0csv, index=False); prog(steps=int(m.num_timesteps), chunks_done=c, last_score=sc)
        log(f'seed {seed} chunk {c}/{NCH} steps {m.num_timesteps} fold0 score {sc:.2f} (Wh/km {row["whkm"]:.2f}, shortfall {row["short"]:.2f} m, unsafe {row["unsafe"]})')
    best = done.loc[done.score.idxmin()]; failed = bool((done.short > 5.0).all())
    tfile = f'{OUT}/sac_seed{seed}_test.csv'
    if not os.path.exists(tfile):
        prog(phase='test'); mb = SAC.load(f'{D}/ckpt_{int(best.steps)}.zip', device='cpu'); R = evaluate(mb, TEST); R.insert(1, 'seed', seed); R.insert(2, 'ckpt_steps', int(best.steps)); R.insert(3, 'gamma', GAMMA); R.to_csv(tfile, index=False)
    json.dump(dict(seed=seed, selected_steps=int(best.steps), fold0_score=float(best.score), failed_to_learn=failed), open(f'{OUT}/sac_seed{seed}_selection.json', 'w'))
    log(f'seed {seed} selected checkpoint {int(best.steps)} steps (fold-0 score {best.score:.2f}); test done' + ('; FAILED-TO-LEARN rule triggered' if failed else '')); prog(phase='done')

def frac(seed):
    pf = f'{CK}/seed{seed}/progress.json'
    if not os.path.exists(pf): return 0.0, 'starting'
    try: st = json.load(open(pf))
    except Exception: return 0.0, 'starting'
    if st.get('phase') == 'done': return 1.0, 'done'
    return 0.9 * st.get('chunks_done', 0) / NCH + (0.05 if st.get('phase') == 'test' else 0.0), f"{st.get('steps', 0) // 1000}k/{NCH * CHUNK // 1000}k" + (' testing' if st.get('phase') == 'test' else '')
if __name__ == '__main__':
    import multiprocessing as mp
    t0 = time.time(); print(f'Safe-SAC (CBF filter, gamma {GAMMA}) at eta 0.65: seeds {SEEDS}, {NCH} x {CHUNK} steps each, started {time.strftime("%H:%M:%S")}'); log('start')
    pre = {s: (os.path.exists(f'{OUT}/sac_seed{s}_test.csv')) for s in SEEDS}
    for s in SEEDS: os.makedirs(f'{CK}/seed{s}', exist_ok=True)
    base = {s: frac(s)[0] for s in SEEDS}
    procs = [mp.Process(target=worker, args=(s,)) for s in SEEDS if not pre[s]]
    for p in procs: p.start()
    try:
        while any(p.is_alive() for p in procs):
            fr = {s: frac(s) for s in SEEDS}; tot = sum(v[0] for v in fr.values()) / len(SEEDS); el = time.time() - t0
            gained = tot - sum(base.values()) / len(SEEDS); left = el * (1 - tot) / gained if gained > 0.01 else 0
            ph = 'Phase 2/2 selecting + testing' if all(v[1] in ('done',) or 'testing' in v[1] for v in fr.values()) else 'Phase 1/2 training'
            wbar = 26; f = int(wbar * tot)
            sys.stdout.write(f"\r{ph} [{'#' * f}{'-' * (wbar - f)}] {int(100 * tot):3d}% | " + ' '.join(f'seed{s}:{v[1]}' for s, v in fr.items()) + f" | elapsed {fmt(el)} left ~{fmt(left)}   "); sys.stdout.flush(); time.sleep(3)
    except KeyboardInterrupt:
        for p in procs: p.terminate()
        raise
    for p in procs: p.join()
    bad = [s for s in SEEDS if not os.path.exists(f'{OUT}/sac_seed{s}_test.csv')]
    msg = f'ALL SEEDS COMPLETE in {fmt(time.time() - t0)}' if not bad else f'FINISHED WITH MISSING SEEDS {bad} - run again to resume'; print('\n' + msg); log(msg)
