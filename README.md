# Tashkent air-quality forecast

A non-commercial research project: forecasts of the city-mean US AQI (from PM2.5) in Tashkent,
Uzbekistan. Not an official service and not medical advice (see `DISCLAIMER.md`). The code is under
the MIT licence (`LICENSE`); no data are included, they belong to their sources.

**Trained models:** [huggingface.co/ProductDmitrii/tashkent-aqi-forecast](https://huggingface.co/ProductDmitrii/tashkent-aqi-forecast)
(LightGBM text files, multi-year norms, model card; CC BY-NC 4.0).

| Forecast | Step | How it is made |
|---|---|---|
| next 24 hours | hour | the hourly model, horizons 1–24 h |
| week | hour | the same model, horizons 1–168 h |
| 3 months | day | days 1–7 from the hourly model, then the norm for the date plus an anomaly decaying over ~4 days |

Documentation:
- `docs/ARCHITECTURE.md`: the whole system (sources, data, models, site, automation) and how to
  set it up from scratch;
- `docs/EXPERIMENTS.md`: every idea tested, with numbers and decisions;
- `research/`: one-off studies outside the service.

## What is and is not in this repository

- **No data.** `data/`, `output/` and `logs/` are in `.gitignore`. The only data file is
  `data/processed/horizon_choice.json`: which model variant serves each horizon group.
- **The target series is not distributed, and no downloader for it is included.** The author's
  copy reads the hourly city-mean US AQI of IQAir sensors from tashkentairdata.uz. To run the code,
  put a series you are allowed to use into `data/raw/tashkentairdata_air.json`; the format is in
  `aqf/sources/tashkentairdata.py`. All other inputs are open and are downloaded by
  `python -m aqf.bootstrap`: the U.S. Embassy monitor (AirNow file archive), Open-Meteo (ERA5,
  archived forecasts, current forecasts) and CAMS through Open-Meteo.
- **The site's live value.** The serverless function `/api/latest` (`aqf/site.py`) reads the URL
  set in the `LATEST_AQI_URL` environment variable of the Vercel project. Without it the page works
  and shows the value as of the forecast run.
- The author's internal notes (work journal, pre-registered checks, scheduler prompts) are not
  included; a few code comments still refer to them.

## Running

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# put the target series into data/raw/tashkentairdata_air.json first (see above)
.venv/bin/python -m aqf.bootstrap          # every other input from scratch (~30–40 min)
.venv/bin/python -m aqf.backtest && .venv/bin/python -m aqf.report
.venv/bin/python -m aqf.forecast           # fresh data + forecast -> output/
.venv/bin/python -m aqf.dashboard          # output/dashboard.html
```

`aqf.forecast` retrains the models when they are more than 7 days old or belong to another model
version (`MODEL_VERSIONS` in `aqf/models.py`; `--retrain` forces it). LightGBM needs OpenMP. On
macOS the author adds an rpath to the OpenMP copy bundled with scikit-learn (macOS strips `DYLD_*`
variables from processes started by system tools); repeat after reinstalling lightgbm:

```bash
install_name_tool -add_rpath @loader_path/../../sklearn/.dylibs .venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib
codesign --force -s - .venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib
```

Results: `output/forecast.json`, `forecast_24h.csv`, `forecast_7d.csv`, `forecast_90d.csv`, and the
page `output/dashboard.html` (one self-contained file, four languages: RU, EN, 中文, Uzbek Cyrillic).

The author publishes the page as a static site on Vercel. `run_daily.sh` (every morning) and
`run_monthly.sh` (the 1st of each month) deploy it through `deploy_site.sh`: build `output/site/`,
smoke tests, `vercel deploy --prod`, smoke tests of the live site. The Vercel project must **not**
be connected to a Git repository (`vercel git disconnect`), or every push publishes the repository
root, that is the code, instead of the site.

Smoke tests (`aqf/smoke.py`, a few seconds) print only problems (`smoke FAIL`, `smoke WARN`) and a
summary line:

- `python -m aqf.smoke build`, before a deploy: 24/168/90 consecutive rows from the forecast origin,
  values and ranges within the scale, LightGBM models no older than 14 days, a plausible backtest
  and 80% ranges, both pages built from the current forecast and their scripts parse
  (`node --check`), no stray files in `output/site/`, `.env*` excluded. A FAIL stops the deploy and
  yesterday's site stays online; a WARN (stale data, a source down, uncommitted code) is reported.
- `python -m aqf.smoke live`, after a deploy: the site serves this very forecast, `/api/latest`
  returns a fresh value, the analytics script is in place, and code and secrets (`/README.md`,
  `/aqf/forecast.py`, `/.env.local`, …) are not served.

Backtest:

```bash
.venv/bin/python -m aqf.backtest --what hourly,daily   # ~20 min
.venv/bin/python -m aqf.report                          # tables + error ranges for the forecast
```

Monthly maintenance (`aqf/monthly.py`): backtest refresh, the 80% ranges checked against real
errors, a report in `logs/monthly_report_<YYYY-MM>.md`, site update.

## Data

- **IQAir city mean** (hourly US AQI, since January 2025): the forecast target.
- **U.S. Embassy monitor** (AirNow, November 2018 – February 2025) extends the history to 7
  winters. Rescaled to the IQAir level (daily values on the common stretch: r = 0.89), shifted by one
  hour (AirNow labels an hour by its start), and its daily profile replaced by the IQAir profile:
  the embassy peaks in the morning, the city mean at 21–22 h.
- **Open-Meteo** weather: the ERA5 archive for training history, the 16-day forecast for daily
  runs, and the archive of past forecasts (since 2024) for honest testing.

## Model

14 models, one per horizon group (1, 2, 3, 4–6, 7–12 … 145–168 h), each with its own variant chosen
on winter 2024/25 (`python -m aqf.select_horizon`, choice in `data/processed/horizon_choice.json`):
1–6 h LightGBM with L1 loss (median), 7–24 h LightGBM with CAMS inputs, 25–168 h a larger LightGBM.
For 4–6 h and 37–48 h the settings come from an evolutionary search (`python -m aqf.tune`, (μ+λ),
49 candidates per horizon, searched on 2024/25): only there were they better on both 2025/26 and
October 2026; elsewhere they overfitted the search winter. Ridge, Extra Trees and an MLP lost to
boosting on every horizon by 1–6 AQI. Holiday and weekend features (`aqf/calendar.py`) were not
chosen on any horizon (day-type effect ±3–4%).

The models predict the change of log(PM2.5) from the last measured hour. Inputs: recent values and
trend; the norm for this hour and date (computed without the row's own season in training); wind,
calm, precipitation, temperature and its daily range, pressure, humidity, cloud, inversion strength
(temperature at 925/850/700 hPa minus 2 m, archive since 2021-03); since 2026-10-28 the westerly flow
over the last 24 h (mean westerly component at the surface and the strongest gust from the west).
Training uses archived weather *forecasts* (since 2024; reanalysis before), so the model learns on
the same kind of inputs it gets in operation. The weather forecast is the mean of four global
models (Open-Meteo best match, GFS, ICON, GEM): its wind error 3–7 days ahead is 10–13% lower than a
single model's.

The chance of a red day (daily mean AQI above 150) for days 1–7 comes from the spread of past
daily-forecast errors (`intervals.json`, `daily_q`) and is shown on the week tiles; from 30% the day
is highlighted. On winters since 2022, 57% of highlighted days were red, and 86% of red days were
highlighted in advance on the first tile (today), 68% on the seventh (`aqf.report.red_week`).
Making the spread depend on the forecast level did not help (Brier 0.112 → 0.116 on 2024/25,
0.098 → 0.101 on 2025/26 + October 2026). For days 8–90 the chance is the share of red days on
those dates in past years: a month's anomaly does not repeat from year to year (correlation −0.05).

Variants compared in `python -m aqf.experiments`; worse or no better than the current model:
weighting recent years (half-life 120/365 days), training on the last 2 years or on IQAir data only,
XGBoost, ensembles, blending with the norm, a "last year + current trend" baseline (41–57 AQI for
the next 24 h), CAMS inputs on all horizons (±0.5 AQI), L1/Huber everywhere, hyper-parameter search,
outlier cleaning of the target, more training rows (all ±0.3).

## Validation

October–March 2020–2026. At the start of every month the model is retrained on all data up to that
day (earlier winters and the start of the current one) and forecasts every 3 hours through the
month. From 2024 on, its input is the archived weather forecast (mean of four models), as in
operation.

Leakage of the future is checked by `python -m aqf.leakage_check`: every value after the forecast
origin is replaced by noise, and the trained model and the forecast must stay identical to the bit
(they do; a planted one-hour leak is caught). Since 2026-11-05 weather forecasts issued after the
origin are replaced by noise too. The parameters of the series merge (embassy scale, daily
profiles) use only data before the origin; smoothing and the outlier filter look only backwards.

Every issued forecast is kept, write-once, with the model version that made it; `python -m aqf.live`
scores them against the measurements (shown on the site; a new version is scored separately while
the table holds forecasts of two versions). That is the only estimate not influenced by model
selection: the variants were chosen on the same winters as in the table below.

Mean absolute error in US AQI points, winter 2025/26 + October 2026, archived weather forecasts as
input (model version 2026-10-28):

| Horizon, h | Model | Persistence | Climatology | Category right |
|---|---|---|---|---|
| 1–3 | 13.4 | 16.0 | 43.6 | 76% |
| 4–6 | 21.5 | 28.9 | 43.6 | 63% |
| 7–12 | 24.7 | 36.8 | 43.7 | 58% |
| 13–24 | 26.6 | 39.8 | 43.7 | 55% |
| 25–48 | 28.7 | 47.1 | 43.8 | 50% |
| 49–72 | 30.8 | 49.8 | 44.0 | 48% |
| 73–120 | 31.9 | 51.8 | 43.9 | 48% |
| 121–168 | 33.3 | 53.2 | 43.9 | 47% |

Averaged over the forecast (winter 2025/26): ±24.2 AQI (24%) for the next 24 hours, ±31.0 (32%) for
the week, ±13.9 (13%) for 1–3 h. By day (all winters 2020–2026): day 1 16.1 against 34.1 for
climatology, days 4–7 24.6 against 33.9. After the second week the forecast equals the norm: nobody
knows the weather that far ahead.

**Neural network blend.** Since model version 2026-11-04 the served forecast from hour 7 on is the
mean (in log PM2.5) of LightGBM and a small neural network (`aqf/nn.py`: a GRU over the last 72 h of
pollution and weather plus a shared decoder for each of the 168 hours ahead); hours 1–6 stay
LightGBM. Against LightGBM version 2026-10-28 on the same (origin, hour) pairs, whole week: 2024/25
32.11 → 31.65 (−0.46, interval −0.78…−0.17), 2025/26 30.95 → 29.92 (−1.03, −1.44…−0.63), October
2026 26.33 → 23.44 (−2.89, −3.90…−1.78). In the 14-forecast shadow run before it went live: week
30.1 → 27.8, 3 days ahead 28.1 → 22.0, first 24 hours 19.1 → 19.9. The network runs as its own
process (`python -m aqf.nn serve`, PyTorch from `requirements-nn.txt` is never imported by the
forecast); if it fails, the forecast goes out from LightGBM alone, stamped with the previous version,
with a warning. Each issued forecast also keeps LightGBM's own forecast for scoring.

## Error breakdown and explanation

**Why a forecast missed** (`python -m aqf.attribution`). The issued daily mean forecast (made in the
morning, 1 or 3 days ahead) is recomputed by the model that issued it and compared with two
variants: *weather* is the recomputation with the forecast weather minus the same with the actual
weather (Open-Meteo archive); *pollution above/below expectations* is the recomputation with actual
weather minus a control run from midnight of that day, also with actual weather; *unknown* is the
control run's own error; *recomputation* is the issued forecast minus its recomputation (about zero
for forecasts that kept their weather and CAMS inputs). The four add up to the whole error. On the
backtest (February 2024 – October 2026, the 06:00 forecast for the next day, mean error ±21 AQI):
weather ±9 (25%), pollution ±12 (32%), unknown ±15 (43%); 3 days ahead 26/37/37%. The parts partly
cancel, so their sizes add up to more than the whole error. The weather column is a *shift*, not a
share: with a perfect weather forecast the error would be 8% lower one day ahead and 12% lower three
days ahead (`research/weather_error_share.py`).

**What the forecast is made of** (`aqf/explain.py`): LightGBM's exact TreeSHAP splits each forecast
hour into the contributions of ~75 inputs, grouped as recent pollution, seasonal norm, temperature
(both weather and the only proxy of heating demand: emissions are not among the inputs), stagnation
(surface and upper wind, calm, inversions, pressure, daily temperature range), precipitation and
humidity, and the rest (time of day, weekday, CAMS). For each day the groups are applied in turn to
the AQI at the origin and averaged over the day's hours; they add up exactly to the day's forecast.
This explains the model, not the physics.

**Drift warning** (`aqf.live.drift`): if over the last 14 days (at least 7 with measurements) the
next-day forecast misses by 15+ AQI on average and 75%+ of days in one direction, the site shows a
warning. On the 2024–2026 backtest the rule fired on 12% of days, about four episodes per winter.
Correcting for recent bias (subtracting the mean error over the last 7/14/28 days, weight 0.25–1)
did not improve any of 24 variants on both test winters and October 2026: biases change faster than
a correction catches up.

**Stagnation or pollution level** (`python -m aqf.stagnation`): for each winter day 2020/21 –
October 2026 (1006 days) the next-day forecast was recomputed from 06:00 with actual weather and its
error compared with the stagnation contribution (SHAP). No relation: error ≈ −3.4 − 0.04 ×
stagnation (r = −0.02). Large monthly misses are level shifts, not stagnation errors.

## Benchmarks and observations (not model inputs)

- **SILAM** (`aqf/silam.py`): the atmospheric-composition model of the Finnish Meteorological
  Institute. FMI publishes its global forecast (20 km, 5 days) in an open S3 bucket but keeps only
  ~32 runs, so each new run is copied for the Tashkent point (PM2.5 and PM10 read directly from the
  remote HDF5 files by HTTP range requests), written once after checks, and compared with our forecast
  of the same day on the same hours. Without city measurements SILAM strongly underestimates Tashkent
  PM2.5 (2–2.4 times below the measurements from the first hours to 5 days ahead, no night peak:
  missing city sources), so a corrected SILAM (scaled by measured/SILAM over the previous 7 days) is
  tracked next to it.
- **Satellites** (`aqf/satellite.py`): daily Sentinel-5P NO₂ over the city and around it (the open
  MEEO copy on AWS, GeoTIFF windows read with rasterio, since 2018) and VIIRS fire detections from
  NASA FIRMS within 200 km. Rank correlation with the daily AQI: NO₂ over the city 0.44 on all days,
  0.52 in winter; fires almost unrelated (−0.13…0.16). Yesterday's NO₂ as a model input was tested
  and rejected (`research/no2_input_test.py`).
- **Emission-cut scenario** (`aqf/scenario.py`): the monthly PM2.5 source shares of the World Bank's
  Air Quality Assessment for Tashkent (2024, fig. 18); a source's contribution shrinks in proportion
  to its emissions. Two scenarios: moderate (heating −30%, dust from outside −5%, transport and heavy
  industry −10%) and strong (−50%, −10%, −25%, −25%). PM2.5 falls by 15–16% in mid-winter months under
  the moderate scenario (27–30% strong); on 8 measured winters there would have been 142 days above
  AQI 150 instead of 211 (96 strong). Kept apart from the forecast.

Also tried and dropped: one model for all horizons (lost to persistence at 1–3 h), daily boosting
for 3 months (worse than the norm after 7 days), quantile calibration of the forecast weather to
ERA5, a heating-season feature (`aqf/heating.py`; within run-to-run noise). The full list is in
`docs/EXPERIMENTS.md`.

## Reliability and security

A failed morning run loses nothing good: the script stops before the deploy, the site keeps the
last good forecast, and the page itself warns when the forecast is more than 30 hours old. Bad
results (NaN, an incomplete forecast, a broken script) are stopped by the smoke tests. Files are
written through a temporary file and replaced whole (`aqf/safe_io.py`). The target cache is
replaced only by a series with the same start that is not shorter; an empty reply of the weather
archive or CAMS does not wipe recent days; a corrupted `models.pkl` falls back to `models_prev.pkl`.
An issued forecast is never overwritten; "days ahead" count from the issue day. A day is the hours
labelled 00:00…23:00 (an hour labelled by its end).

The page never inserts HTML from data (text only), and every "<" in embedded data is escaped.
Headers (`aqf/site.py`): a Content-Security-Policy with scripts only from the site itself and inline
scripts by hash (recomputed at every build and checked by the smoke test), styles and fonts from
Google, connections only back to the site, no framing; nosniff, X-Frame-Options, Referrer-Policy,
Permissions-Policy. `/api/latest` keeps the upstream reply in memory for 30 s and the refresh
button's request is cached at the Vercel edge for 30 s, so cache-busting cannot turn into a flood of
upstream requests; the upstream value is validated. Code, data and `.env*` are not deployed (checked
by the smoke test).

## Limitations

- A forecast for the city as a whole (the mean over IQAir sensors), not for a particular station.
- In the backtest the inversion features come from an archive of near-analyses (there is no archive
  of upper-air forecasts), so the backtest is slightly optimistic with them beyond one day.
- Errors of the weather forecast, sudden episodes (dust, fires, abrupt inversions) and level shifts
  lasting weeks are the main limits; see `docs/EXPERIMENTS.md`.

## Licence and disclaimer

Code and documentation: MIT (`LICENSE`), "as is", without warranty. Trained models: CC BY-NC 4.0 on
Hugging Face. No data are included; third-party data (IQAir via tashkentairdata.uz, the U.S. Embassy
monitor via AirNow, Open-Meteo, Copernicus CAMS and Sentinel-5P, FMI SILAM, NASA FIRMS) are not
covered by either licence and are used under their owners' terms. What the project does not
promise: `DISCLAIMER.md`.
