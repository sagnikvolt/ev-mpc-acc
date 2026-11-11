# BMW i3 (60 Ah) data check and first physics fit — Sonnet, 2026-11-08
Source: TU Munich, IEEE DataPort DOI 10.21227/6jr9-5235, CC-BY. Files: `data/raw/bmw_i3/Measurement Data.zip` (+ readme.txt), unzipped to `data/raw/bmw_i3/measurement/`.

## Data facts (verified)
- **70 trips** (32 summer "A", 38 winter "B"), not 72 (72 = 70 CSVs + Overview.xlsx + readin.m). 1.09M rows, 1,358 km, 30.4 h.
- **10 Hz exactly** (dt = 0.1 s, no gaps). Delimiter `;`, latin-1 encoding. Elevation is integer metres (grade estimate is noisy).
- **Current sign: negative = discharging.** Use P_batt = −U·I for consumption.
- TripB38 has a header typo (`Velocity [km/h]]]`); match columns by prefix. Category A is missing some channels (readme).
- **TripB04 and TripB09 contain charging** (SoC rises / current positive): excluded from energy fits.
- Energy from U·I: median **164 Wh/km** overall (A: 133, B: 190 with heating).
- **SoC scale is consistent here:** 23.1 kWh per 100 % SoC (IQR 22.7–23.4, corr 0.95, 42 trips). This car is usable for absolute battery energy, unlike Zoe.

## First physics fit (`python/fit_bmw_i3.py`, trip-wise 70/30 split, seed 42, mass ASSUMED 1270 kg)
| Model | Train median trip error | Test median trip error | Test pooled error |
|---|---|---|---|
| v1 constant efficiencies | 11.3 % | 9.6 % | −7.9 % |
| v2 load-dependent losses | 10.2 % | 9.8 % | −4.4 % |
**Gate 2 (< ~5 % trip energy error) NOT yet passed.** Fit hits parameter bounds (regen efficiency ≥ 1, quadratic loss → 0): a sign the structure, not the data volume, is the problem. Mass and grade changes made no difference (checked: free mass, no grade).

## Why: auxiliary load
Standstill battery power (≈ auxiliary load) ranges **0.31–6.39 kW** across trips (A mean 1.07, B mean 2.67 kW). Average traction power is only ~8 kW, so a ±1 kW aux error alone is ~12 % trip error. Heating power signals correlate only ~0.5–0.6 with standstill power (cycling heater, pumps, 12 V loads not measured).

## Proposed next steps (for next Sonnet session, no new data needed)
1. Fit aux explicitly: a0 + a1·heating(CAN) + a2·AirCon, per category A/B, plus ambient-temperature term.
2. Report two numbers: **traction-only energy** (subtract fitted aux) and **total**. Gate 2 on traction-only is defensible for an ACC paper, since ACC cannot change the heater; state this clearly.
3. Use measured motor torque and speed (channels exist) to build a motor-efficiency map instead of one quadratic.
4. Validate the 1270 kg mass assumption (check i3 spec sheet / TUM paper in MDPI WEVJ 11(2):41).

---
# Update 2 (Sonnet, same day): explicit auxiliary model + 5-fold trip-wise CV
Scripts: `python/fit_bmw_i3_v3.py`, `python/cv_bmw_i3.py` (outputs `results/bmw_i3_fit_trips_v3.csv`, `results/bmw_i3_cv_trips.csv`).
Model: road-load physics (mass ASSUMED 1270 kg, CdA & Crr fitted) → load-dependent traction loss (quadratic) + aux = b0 + b1·heatCAN + b2·AirCon + b3·max(0,15−Tamb).

| Check (trip-wise, no trip in both train and test) | Result |
|---|---|
| Single 70/30 split, train / test median trip error | 5.3 % / 7.5 % (was ~10 % before aux model) |
| **5-fold CV, 68 trips: median / mean / 90th pct abs trip error** | **5.8 % / 9.4 % / 15.5 %** |
| 5-fold CV pooled energy error (sum over all trips) | **0.76 %** |
| Share of trips within 5 % / 10 % | 44 % / 72 % |
| Winter (B, 36 trips) / summer (A, 32 trips) median abs error | 3.8 % / 8.1 % |
| By length: <8 km / 8–15 / 15–25 / >25 km | 19 % / 4.1 % / 6.9 % / 4.8 % |
Worst trips: A06 (+103 %, 46 km, only 5.0 kWh measured, model predicts ~2×: suspect data/elevation problem, inspect), B13 (2.8 km), A01/A23/A14 (short, summer).

## Honest reading
- **Gate 2 target (< ~5 % per-trip) is narrowly missed on the median (5.8 %) but met on pooled energy (0.76 %) and in winter (3.8 %).** Short trips inflate relative error because the denominator is tiny; report Wh/km MAE as well.
- Summer category A is weaker: the readme says A lacks some channels (measurement trouble), so aux estimation there is poorer.
- Fitted regen coefficient >1 (r1≈1.22) is physically odd. Likely absorbs the unverified mass/accel-smoothing assumptions. Do not publish efficiency numbers until mass is verified.
- Torque-based traction model (k·T·v) failed (params at bounds, RMSE 9 kW): the "Motor Torque" channel is not directly wheel-side torque in the way assumed. Dropped, not used.
- **Recommendation to Opus for the Gate 2 decision:** treat physics-only (median 5.8 %, pooled 0.8 %) as the *baseline*; the ML residual is what should push per-trip error below 5 % — that is exactly the paper's claim, and the Gate 3 test (≥30 % RMSE cut on held-out trips) measures it. Do NOT tune physics further to hit 5 %; it would hide the benefit of the ML part.
