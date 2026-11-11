"""AWS Lambda handler: runs the paper's real energy model, conventional ACC and battery-aware MPC on an uploaded lead-vehicle speed trace.
No science code is modified: traces are recorded by wrapping the ACC policy and subclassing MPC.solve.

Request (POST /simulate, JSON):
  speeds   : list of lead-vehicle speeds
  hz       : sampling rate of `speeds` in Hz (1, 2, 5 or 10; default 10)
  unit     : "mps" (default) or "kmh"
  eta      : regen efficiency 0.3-1.0 (default 0.65)
  cap_kw   : regen power cap 10-100 kW (default 35)
  preview  : "oracle" (MPC knows the lead's future: the ceiling) or "const" (naive hold-speed preview); default "oracle"
  breakers : optional list of speed-breaker positions in metres (switches on the map-aware arms and makes the lead slow for them, as in paper experiment S1b)
Limits: 30-180 s of lead trace, speeds 0-46 m/s (ego car capped at 38 m/s), at most 20 breakers.
"""
import os, sys, json, math, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = next((c for c in (HERE, os.path.abspath(os.path.join(HERE, '..', '..'))) if os.path.exists(os.path.join(c, 'code', 'python', 'mpc_acc.py'))), None)
if ROOT is None: raise RuntimeError('package root (code/python/mpc_acc.py) not found')

# mpc_acc reads ~/mnt/EV Project/results/bmw_i3_params_final.npy at import: build it from the packaged CSV under a scratch HOME.
import numpy as np
_tmp = tempfile.mkdtemp(prefix='evdemo_')
os.makedirs(os.path.join(_tmp, 'mnt', 'EV Project', 'results'))
_vals = [float(l.split(',')[1]) for l in open(os.path.join(ROOT, 'derived_data', 'bmw_i3_params_final.csv')).read().strip().splitlines()[1:]]
np.save(os.path.join(_tmp, 'mnt', 'EV Project', 'results', 'bmw_i3_params_final.npy'), np.array(_vals))
os.environ['HOME'] = _tmp
sys.path.insert(0, os.path.join(ROOT, 'code', 'python'))
import mpc_acc as A, baselines as B

MIN_S, MAX_S, VMAX = 30, 180, 46.0   # lead may be fast (real traces reach ~46 m/s); the ego car is still capped at 38 m/s inside the model
HEADERS = {'Content-Type': 'application/json'}
class BadRequest(Exception): pass
def _resp(code, body): return {'statusCode': code, 'headers': HEADERS, 'body': json.dumps(body)}

def _parse(ev):
    try: req = json.loads(ev['body']) if isinstance(ev.get('body'), str) else (ev.get('body') or ev)
    except Exception: raise BadRequest('body is not valid JSON')
    if not isinstance(req, dict): raise BadRequest('body must be a JSON object')
    sp = req.get('speeds')
    if not isinstance(sp, list) or not sp: raise BadRequest('"speeds" must be a non-empty list')
    try: x = np.array(sp, dtype=float)
    except Exception: raise BadRequest('"speeds" must be numbers')
    if not np.all(np.isfinite(x)): raise BadRequest('"speeds" contains non-numeric values')
    hz = req.get('hz', 10)
    if hz not in (1, 2, 5, 10): raise BadRequest('"hz" must be 1, 2, 5 or 10')
    unit = req.get('unit', 'mps')
    if unit not in ('mps', 'kmh'): raise BadRequest('"unit" must be mps or kmh')
    if unit == 'kmh': x = x / 3.6
    dur = (len(x) - 1) / hz
    if dur < MIN_S: raise BadRequest(f'trace is {dur:.0f} s long; need at least {MIN_S} s')
    if dur > MAX_S: raise BadRequest(f'trace is {dur:.0f} s long; the demo allows at most {MAX_S} s')
    if x.min() < -0.5 or x.max() > VMAX: raise BadRequest(f'speeds must be between 0 and {VMAX:.0f} m/s ({VMAX * 3.6:.0f} km/h)')
    t = np.arange(len(x)) / hz; t10 = np.arange(0, t[-1] + 1e-9, 0.1); vl = np.interp(t10, t, x)
    try: eta = float(req.get('eta', 0.65)); cap = float(req.get('cap_kw', 35.0))
    except Exception: raise BadRequest('"eta" and "cap_kw" must be numbers')
    if not 0.3 <= eta <= 1.0: raise BadRequest('"eta" must be 0.3-1.0')
    if not 10 <= cap <= 100: raise BadRequest('"cap_kw" must be 10-100')
    prev = req.get('preview', 'oracle')
    if prev not in ('oracle', 'const'): raise BadRequest('"preview" must be oracle or const')
    bk = req.get('breakers') or []
    if not isinstance(bk, list) or len(bk) > 20: raise BadRequest('"breakers" must be a list of at most 20 positions')
    try: bk = sorted(float(b) for b in bk)
    except Exception: raise BadRequest('"breakers" must be numbers (metres)')
    if any(b < 0 or b > 20000 for b in bk): raise BadRequest('breaker positions must be 0-20000 m')
    return vl, eta, cap, prev, bk

def lead_slows_for_breakers(vl, bk, vb=4.2, a_dec=2.0, a_acc=1.0, dt=0.1):
    """Same lead-vehicle model as paper experiment S1b (scenarios.s1_breakers) but with user-given breaker positions (metres, ego frame)."""
    out = np.empty_like(vl); xl = 5.0 + 1.5 * vl[0]; prev = vl[0]
    for i in range(len(vl)):
        cap = min((np.sqrt(vb ** 2 + 2 * a_dec * (xb - xl)) if xl <= xb else np.sqrt(vb ** 2 + 2 * a_acc * (xl - xb))) for xb in bk)
        tgt = min(vl[i], cap); prev = float(np.clip(tgt, prev - 3.0 * dt, prev + 2.0 * dt)) if i else vl[0]; out[i] = prev; xl += prev * dt
    return out

class TraceMPC(A.MPC):
    """Same controller; just records state before each solve."""
    def __init__(s, *a, **k): super().__init__(*a, **k); s.rec = []
    def solve(s, v, d, pv, a_prev, **kw):
        ok, a = super().solve(v, d, pv, a_prev, **kw); s.rec.append((float(v), float(d), float(a))); return ok, a

def _slim(m):
    keys = ('whkm_adj', 'whkm', 'E_wh', 'dist_m', 'dmin', 'gap_mean', 'gap_err_rms', 'jerk_rms', 'viol', 'short_max', 'steps', 'ncross', 'cross_excess', 'fail')
    out = {}
    for k in keys:
        if k in m:
            f = float(m[k]); out[k] = f if math.isfinite(f) else None
    return out

def run(vl, eta, cap, prev, bk):
    A.set_regen(eta, cap=cap)
    if bk: vl = lead_slows_for_breakers(vl, bk)   # the lead brakes for breakers, as in S1b
    rec_acc = []; base = B.acc_map_policy(0.4, 1.0) if bk else B.acc_policy(0.4, 1.0)
    def pol(s): a = base(s); rec_acc.append((float(s['v']), float(s['d']), float(a))); return a
    mpc = TraceMPC()
    if bk:
        acc = B.simulate(vl, pol, breakers=bk)
        mp = A.simulate(vl, mpc, preview='constmap' if prev == 'const' else 'oracle', use_map=True, breakers=bk)
    else:
        acc = B.simulate(vl, pol); mp = A.simulate(vl, mpc, preview=prev)
    n = min(len(rec_acc), len(mpc.rec))
    r = lambda rows, j: [round(rw[j], 3) for rw in rows[:n]]
    d = (mp['whkm_adj'] / acc['whkm_adj'] - 1.0) * 100.0
    return dict(acc=_slim(acc), mpc=_slim(mp), energy_change_pct=round(d, 3),
                trace=dict(t=[round(0.5 * k, 1) for k in range(n)], lead=[round(float(vl[5 * k]), 3) for k in range(n)], acc_v=r(rec_acc, 0), acc_gap=r(rec_acc, 1), mpc_v=r(mpc.rec, 0), mpc_gap=r(mpc.rec, 1)),
                settings=dict(eta=eta, cap_kw=cap, preview=prev, breakers=bk, seconds_simulated=n * 0.5))

def lambda_handler(event, context=None):
    if (event.get('requestContext', {}).get('http', {}).get('method') or event.get('httpMethod')) == 'OPTIONS': return _resp(200, {})
    try: vl, eta, cap, prev, bk = _parse(event)
    except BadRequest as e: return _resp(400, {'error': str(e)})
    try: return _resp(200, run(vl, eta, cap, prev, bk))
    except Exception as e:
        print('simulation error', repr(e)); return _resp(500, {'error': 'simulation failed'})
