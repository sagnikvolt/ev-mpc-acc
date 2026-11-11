#!/usr/bin/env python3
"""Clean the Renault Zoe time series (Zenodo 10.5281/zenodo.7033914) and split it into trips.

Usage:  python3 -I python/clean_zoe.py <raw_csv_dir> <out_dir>
  raw_csv_dir : folder containing the two unzipped ZOE_CSV_*.csv files (never modified)
  out_dir     : data/processed/

Outputs (all numbers in the QC report come from this script):
  zoe_trips.csv         one row per second of driving, only rows inside accepted trips
  zoe_trip_summary.csv  one row per accepted trip (distance, energy, Wh/km, split)
  zoe_qc_report.md      counts of every filter applied
Seed for the train/val/test trip split: 42.
"""
import sys, os, glob
import numpy as np, pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]
SEED = 42
GAP_S = 10            # a gap longer than this between speed samples starts a new trip
IDLE_S = 300          # a standstill longer than this ends a trip (idle rows dropped)
MIN_DUR_S, MIN_KM = 120, 0.5
MIN_COVERAGE = 0.95   # energy is only computed for trips with at least this share of valid power samples
COLS = ["date","time","avail_kwh","cons_kw","rpm","torque_nm","odo_km","speed_kmh",
        "soc","throttle","avail_chg_kw","chg_kw"]
qc = {}

# ---- 1. read: keep rows where vehicle speed is present ---------------------------------
files = sorted(glob.glob(os.path.join(RAW, "ZOE_CSV_*.csv")))
parts, raw_rows = [], 0
for f in files:
    n_file = 0
    for ch in pd.read_csv(f, sep=";", skiprows=2, names=COLS, na_values=["ND"],
                          decimal=",", chunksize=2_000_000):
        n_file += len(ch)
        parts.append(ch[ch.speed_kmh.notna()])
    qc[f"raw rows in {os.path.basename(f)}"] = n_file
    raw_rows += n_file
d = pd.concat(parts, ignore_index=True)
qc["raw rows total"] = raw_rows
qc["rows with speed present"] = len(d)
d["t"] = pd.to_datetime(d["date"] + " " + d["time"], format="%d/%m/%Y %H:%M:%S")
d = d.drop(columns=["date", "time"]).sort_values("t").drop_duplicates("t").reset_index(drop=True)
qc["rows after removing duplicate timestamps"] = len(d)

# ---- 2. physically impossible values -> NaN (kept row, flagged) -------------------------
bad = {"soc>100": d.soc > 100, "soc<=0": d.soc <= 0,
       "throttle>100": d.throttle > 100, "cons_kw>110": d.cons_kw > 110}
d["qc_flag"] = 0
for i, (k, m) in enumerate(bad.items()):
    qc[f"values set to NaN: {k}"] = int(m.sum())
    d.loc[m, "qc_flag"] |= (1 << i)
d.loc[bad["soc>100"] | bad["soc<=0"], "soc"] = np.nan
d.loc[bad["throttle>100"], "throttle"] = np.nan
d.loc[bad["cons_kw>110"], "cons_kw"] = np.nan
# rpm is ~41 x speed (rescaled copy of speed); flag spikes, do not alter
rpm_spike = (d.speed_kmh > 20) & ((d.rpm - 41 * d.speed_kmh).abs() > 0.2 * 41 * d.speed_kmh)
d["rpm_spike"] = rpm_spike.astype(int)
qc["rpm spikes flagged (|rpm-41*speed|>20%, speed>20)"] = int(rpm_spike.sum())

# ---- 3. trips -----------------------------------------------------------------------------
dt = d.t.diff().dt.total_seconds()
new_gap = (dt > GAP_S) | dt.isna()
# long standstills: mark rows in a zero-speed run longer than IDLE_S
z = d.speed_kmh == 0
run_id = ((z != z.shift()) | new_gap).cumsum()
run_len = z.groupby(run_id).transform("size")
long_idle = z & (run_len > IDLE_S)
qc["rows dropped inside standstills > 300 s"] = int(long_idle.sum())
d = d[~long_idle].copy()
dt = d.t.diff().dt.total_seconds()
d["seg"] = ((dt > GAP_S) | dt.isna()).cumsum()
# also split where a dropped idle run sat between two rows (time jump > GAP_S already covers it)
qc["segments before filtering"] = int(d.seg.nunique())

g = d.groupby("seg")
dur = g.t.agg(lambda s: (s.max() - s.min()).total_seconds())
km = g.speed_kmh.sum() / 3600.0          # 1 Hz, so sum(km/h)/3600 = km
keep = dur[(dur >= MIN_DUR_S) & (km >= MIN_KM)].index
qc[f"segments with duration < {MIN_DUR_S}s or distance < {MIN_KM} km (dropped)"] = int(d.seg.nunique() - len(keep))
d = d[d.seg.isin(keep)].copy()
d["trip_id"] = pd.factorize(d.seg)[0] + 1
d = d.drop(columns="seg")

# ---- 4. derived signals -------------------------------------------------------------------
d["speed_ms"] = d.speed_kmh / 3.6
d["time_in_trip_s"] = d.groupby("trip_id").t.transform(lambda s: (s - s.iloc[0]).dt.total_seconds())
def accel(s, t):
    v = s.values; tt = t.values
    a = np.empty_like(v); 
    a[1:-1] = (v[2:] - v[:-2]) / (tt[2:] - tt[:-2])
    a[0] = (v[1] - v[0]) / (tt[1] - tt[0]) if len(v) > 1 else 0.0
    a[-1] = (v[-1] - v[-2]) / (tt[-1] - tt[-2]) if len(v) > 1 else 0.0
    return pd.Series(a, index=s.index)
d["accel_ms2"] = d.groupby("trip_id", group_keys=False).apply(lambda x: accel(x.speed_ms, x.time_in_trip_s), include_groups=False)
d["dist_m"] = d.groupby("trip_id").speed_ms.cumsum()
d["month"] = d.t.dt.month

# ---- 5. trip summary + trip-wise split ------------------------------------------------------
def summ(x):
    n = len(x); dkm = x.speed_ms.sum() / 1000.0
    covered = x.cons_kw.notna().mean()
    e = x.cons_kw.sum() / 3600.0 if covered >= MIN_COVERAGE else np.nan   # kWh, 1 Hz rectangle rule
    return pd.Series({"start": x.t.iloc[0], "duration_s": n, "distance_km": dkm,
        "mean_speed_kmh": x.speed_kmh.mean(), "max_speed_kmh": x.speed_kmh.max(),
        "stop_fraction": (x.speed_kmh == 0).mean(),
        "energy_kwh": e, "wh_per_km": 1000 * e / dkm if (dkm > 0 and not np.isnan(e)) else np.nan,
        "cons_coverage": covered,
        "soc_start": x.soc.dropna().iloc[0] if x.soc.notna().any() else np.nan,
        "soc_end": x.soc.dropna().iloc[-1] if x.soc.notna().any() else np.nan,
        "avail_start_kwh": x.avail_kwh.dropna().iloc[0] if x.avail_kwh.notna().any() else np.nan,
        "avail_end_kwh": x.avail_kwh.dropna().iloc[-1] if x.avail_kwh.notna().any() else np.nan})
ts = d.groupby("trip_id").apply(summ, include_groups=False).reset_index()
rng = np.random.default_rng(SEED)
perm = rng.permutation(ts.trip_id.values)
n = len(perm); n_tr, n_va = int(0.70 * n), int(0.15 * n)
split = {**{i: "train" for i in perm[:n_tr]}, **{i: "val" for i in perm[n_tr:n_tr + n_va]},
         **{i: "test" for i in perm[n_tr + n_va:]}}
ts["split"] = ts.trip_id.map(split)
d["split"] = d.trip_id.map(split)
qc["accepted trips"] = int(n)
qc["accepted driving rows"] = int(len(d))
qc["accepted distance (km)"] = round(float(ts.distance_km.sum()), 1)
qc["accepted driving time (h)"] = round(float(ts.duration_s.sum() / 3600), 1)
qc["trips per split (train/val/test)"] = ts.split.value_counts().reindex(["train","val","test"]).tolist()
qc["rows per split (train/val/test)"] = d.split.value_counts().reindex(["train","val","test"]).tolist()

# ---- 6. write -------------------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
out_cols = ["trip_id","split","t","time_in_trip_s","speed_kmh","speed_ms","accel_ms2","dist_m","odo_km",
            "cons_kw","torque_nm","rpm","rpm_spike","throttle","soc","avail_kwh","avail_chg_kw","chg_kw","month","qc_flag"]
d[out_cols].to_csv(os.path.join(OUT, "zoe_trips.csv"), index=False, float_format="%.5g")
ts.to_csv(os.path.join(OUT, "zoe_trip_summary.csv"), index=False, float_format="%.5g")

whkm = ts.wh_per_km.dropna()   # only trips with enough power coverage
with open(os.path.join(OUT, "zoe_qc_report.md"), "w", encoding="utf-8") as fh:
    fh.write("# Zoe cleaning report (generated by python/clean_zoe.py)\n\n")
    fh.write(f"Parameters: GAP_S={GAP_S}, IDLE_S={IDLE_S}, MIN_DUR_S={MIN_DUR_S}, MIN_KM={MIN_KM}, split seed={SEED}\n\n")
    fh.write("| Item | Value |\n|---|---|\n")
    for k, v in qc.items(): fh.write(f"| {k} | {v} |\n")
    fh.write("\n## Wh/km per accepted trip (battery energy = sum of 'Consumption (kW)' x 1 s / distance)\n\n")
    fh.write(whkm.describe(percentiles=[.05,.25,.5,.75,.95]).round(1).to_frame("Wh/km").to_markdown() + "\n")
    # cross-checks of the energy integral (computed here, not assumed)
    ok = ts.dropna(subset=["energy_kwh"])
    qc2 = {"trips with energy computed (power coverage >= %.2f)" % MIN_COVERAGE: len(ok),
           "trips without energy (coverage too low)": int(len(ts) - len(ok)),
           "trips with negative net energy (net regeneration)": int((ok.energy_kwh < 0).sum())}
    g = ok[(ok.distance_km >= 5)].copy(); g["soc_drop"] = g.soc_start - g.soc_end; g = g[g.soc_drop > 0.5]
    if len(g) > 5:
        a, b = np.polyfit(g.soc_drop, g.energy_kwh, 1)
        qc2["cross-check A: trips >=5 km with SOC drop"] = len(g)
        qc2["cross-check A: corr(energy integral, SOC drop)"] = round(float(np.corrcoef(g.soc_drop, g.energy_kwh)[0, 1]), 3)
        qc2["cross-check A: implied kWh per 100% SOC (UNRESOLVED: car reports ~51 kWh available)"] = round(float(a * 100), 1)
    fh.write("\n## Cross-checks and open issues\n\n| Item | Value |\n|---|---|\n")
    for k, v in qc2.items(): fh.write(f"| {k} | {v} |\n")
    fh.write("""
Notes
- The 2019 file (`ZOE_CSV_2019-01-01_2020-01-19.csv`) only contains 19/01/2020 at 1 reading/minute, so its 143 speed rows never form a trip.
- Speed is logged at 1 Hz during driving (not 5 s). Motor RPM is about 41 x speed (a rescaled copy of speed): redundant, do not treat as an independent feature.
- No ambient temperature, road grade or GPS in this dataset.
- Absolute scale of the SOC signal vs the energy integral is unresolved (see cross-check A). Use the Zoe for relative trends and for the unseen-car test, not for absolute battery calibration.
- 'Available_Energy' jumps at trip boundaries and is not used as a trip-level energy check.
""")
    fh.write("\n## Trip summary statistics\n\n")
    fh.write(ts[["duration_s","distance_km","mean_speed_kmh","max_speed_kmh","stop_fraction","cons_coverage"]]
             .describe().round(2).to_markdown() + "\n")
print(open(os.path.join(OUT, "zoe_qc_report.md"), encoding="utf-8").read())
