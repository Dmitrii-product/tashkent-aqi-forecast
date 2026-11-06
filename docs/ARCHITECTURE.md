# Architecture: data, models, site, automation

The whole system, so that it can be understood and set up again from scratch. Every idea tested,
with numbers: `docs/EXPERIMENTS.md`; accuracy and method: `README.md`.

## 1. Overview

```
 sources                          cache (data/, not in git)            computation (aqf/)
 ───────                          ─────────────────────────            ──────────────────
 IQAir city mean (supplied) ──┐   raw/tashkentairdata_air.json
 AirNow, U.S. Embassy ────────┤   raw/airnow/*.parquet           ┌─ data.py: one hourly PM2.5 series
 Open-Meteo: ERA5 archive ────┼─► raw/weather/archive.parquet ───┤   (IQAir scale) + weather
   upper-air temperatures ────┤   raw/weather/upper.parquet      ├─ models.py: 14 models by horizon
   archived forecasts (4) ────┤   raw/weather/prevrun_day*.parquet   + 90 days (LongRange)
   16-day forecast ───────────┤   raw/weather/models/*_day*.parquet ├─ forecast.py: daily forecast
 CAMS (via Open-Meteo) ───────┘   raw/weather/cams.parquet        └─► output/forecast.json
                                                                        data/forecast_log/<time>.json
 checks and analyses: backtest → report → (intervals, backtest_summary) ─┐
 live (real use), attribution, explain, stagnation ──────────────────────┤
 nn.py: neural network (blended from hour 7, blend.py) → <time>_nn.json ──┤
                                                                          ▼
 dashboard.py + dashboard_template.html ─► output/dashboard.html (self-contained page)
 site.py ─► output/site/{index.html, api/latest.js, vercel.json} ─► Vercel
 smoke.py: checks before and after a deploy (deploy_site.sh)
 scheduler: run_daily.sh (~07:15 Tashkent time), run_monthly.sh (the 1st, 09:00)
```

## 2. Layout

| Path | What | In git |
|---|---|---|
| `aqf/` | package: data, models, forecast, checks, site | yes |
| `aqf/sources/` | source loaders | yes |
| `aqf/dashboard_template.html` | the page: markup, styles, script, texts in 4 languages (`I18N`) | yes |
| `data/processed/horizon_choice.json` | which model variant serves each horizon group | yes (the only data file in git) |
| `research/` | one-off studies outside the service | yes |
| `docs/` | this documentation | yes |
| `run_daily.sh`, `run_monthly.sh`, `deploy_site.sh` | daily run, monthly run, deploy | yes |
| `requirements.txt` / `requirements-nn.txt` | dependencies of the service / of the neural network (PyTorch) | yes |
| `data/raw/` | source cache | no |
| `data/processed/` | models (`models.pkl`, `models_prev.pkl`), backtests, ranges, analyses | no |
| `data/forecast_log/` | every issued forecast (with its model version) + the weather and CAMS it was made with + the network's part (`_nn.json`) | no |
| `data/silam/<YYYYMMDD>.parquet` | SILAM runs for Tashkent (write-once) | no |
| `data/satellite/no2/`, `data/satellite/firms/` | Sentinel-5P NO₂ days, FIRMS fire days and yearly archives (write-once) | no |
| `output/` | `forecast.json`, CSV files, `dashboard.html`, `site/` (+ `.vercel`, `.env.local`) | no |
| `output/model_export/<name>/` | the copy of the models for publication (`aqf/export.py`) | no |
| `logs/` | run logs, monthly reports | no |
| `tools/` | a local Vercel CLI (if there is no global one) | no |

## 3. Data collection

| Source | Module | What | Notes |
|---|---|---|---|
| IQAir city mean (author: tashkentairdata.uz) | `sources/tashkentairdata.py` | hourly US AQI since 2025-01-05 (`start` + `values`, UTC+5) | The target. Not distributed and not downloaded by this code: supply the file yourself |
| AirNow (U.S. Embassy) | `sources/airnow.py` | hourly PM2.5, 2018-11 – 2025-02 | Files `HourlyData_YYYYMMDDHH.dat`; station UZB010001 is at the end of each file, so a range request for the last 4 KB is enough (~25 min for the whole history). AirNow labels an hour by its start, IQAir by its end: shift +1 h |
| Open-Meteo archive | `sources/openmeteo.py` | ERA5: temperature, humidity, dew point, precipitation, snow, cloud, pressure, wind, gusts, radiation | The past, and the "actual weather" in analyses |
| Open-Meteo historical-forecast | `sources/openmeteo.py` | temperature at 925/850/700 hPa, wind at 850 hPa (since 2021-03) | Inversion strength |
| Open-Meteo previous-runs | `sources/openmeteo.py` | archived weather forecasts by lead time (1–7 days) since 2024-01 for best match, GFS, ICON, GEM | The model trains on them; HTTP 429 → 65 s pause, 2 s between requests |
| Open-Meteo forecast | `sources/openmeteo.py` | 16-day forecast, 4 models + upper levels | In operation: the mean of the four models (`data.ensemble_mean`, wind direction averaged as a vector) |
| CAMS via Open-Meteo air-quality | `sources/cams.py` | PM2.5, PM10, dust: a merged series since 2022-08 + 5-day forecast | Inputs for horizons ≤ 24 h only |
| SILAM (FMI), public S3 bucket `fmi-opendata-silam-surface-netcdf` | `silam.py` | 5-day PM2.5 and PM10 forecast at the Tashkent point and a 3×3 square, since 2026-09-26 | Not a model input but a benchmark. FMI keeps ~32 runs, so each new one is copied at once: HDF5 read by HTTP range requests (h5py), ~2 min per run; written once after checks (10 files, 120 consecutive hours, a point near Tashkent, 0–3000 µg/m³, PM10 ≥ PM2.5); a failed run is retried the next day. Licence CC BY 4.0 |
| Sentinel-5P/TROPOMI NO₂, MEEO copy on AWS Open Data (`s3://meeo-s5p`, no account) | `satellite.py` | tropospheric NO₂ column and qa_value of each overpass (GeoTIFF, 0.035° grid): mean over the city (≤ 15 km) and in a 30–50 km ring, qa ≥ 75, since 2018 | Not a model input. A 1.4°×1° window read by HTTP range requests (rasterio), overpasses 07:00–10:30 UTC only; OFFL (final, ~9 days later) and NRTI (same day) kept apart, OFFL preferred |
| NASA FIRMS, VIIRS Suomi NPP | `satellite.py` | fire detections within 200 km: the open 7-day file of the Russia_Asia region and yearly country archives (Uzbekistan, Kazakhstan, Tajikistan, Kyrgyzstan; 2018–2025) | NASA publishes a year's archive the following year. Persistent heat sources (archive type 2 and 0.01° cells lit on 30+ days) are counted apart from fires |

First full download: `python -m aqf.bootstrap` (~30–40 min). Daily update: `forecast.refresh_data()`,
each source on its own; a failure of one does not stop the forecast (a warning, the cache is used).
Empty replies are refused; files are written atomically (`aqf/safe_io.py`).

## 4. Target series (`aqf/data.py`)

One hourly PM2.5 series on the IQAir scale: the embassy monitor until 2025-01-05, IQAir after.
- The embassy scale is mapped to IQAir on the common stretch (`calibrate`, only data before the
  forecast origin; fewer than 20 common days, no correction).
- The embassy's daily profile is replaced by the IQAir profile (`harmonize_diurnal`): the embassy
  peaks in the morning, the city mean at 21–22 h.
- Isolated spikes are damped by a backward-looking filter (`smooth_spikes`, `clean_embassy`); light
  smoothing is backward-looking too (`EMB_SMOOTH = 0.3`).
- `load_target(profile_cutoff=…)` builds the series as it was known at the cutoff, so the backtest
  does not see the future.
- A day is the hours labelled 00:00…23:00 (an hour labelled by its end).

## 5. Models (`aqf/models.py`, `aqf/features.py`)

**Hourly model (24 hours and week).** Direct multi-horizon forecast: 14 horizon groups (1, 2, 3,
4–6, 7–12, 13–18, 19–24, 25–36, 37–48, 49–72, 73–96, 97–120, 121–144, 145–168 h), each with its own
LightGBM booster. `HorizonRouter` gives each group to the variant in
`data/processed/horizon_choice.json` (chosen on winter 2024/25): L1 for 1–3 h and 97–120 h, settings
from the evolutionary search for 4–6 h and 37–48 h, the CAMS variant for 7–24 h, a larger LightGBM
elsewhere. The target is the change of log1p(PM2.5) from the last measured hour (`lag0`).

Features of a row (origin, h), about 75 (`features.hourly_rows`):
- pollution: lags 0–48 h, means over 3–168 h, daily range and spread, trends, the same hour on past
  days;
- seasonal norm: climatology by (day of year, hour), in training computed without the row's own
  season (`Climatology`; otherwise a leak), and anomalies from it;
- weather at hour T (the forecast with the right lead time): temperature, humidity, wind and its
  components, gusts, calm, precipitation over 24/72 h, snow, cloud, radiation, pressure and its
  tendency, heating degree days, daily temperature range, inversions 925/850/700 hPa, wind at 850 hPa,
  westerly flow over the 24 h before T (mean westerly surface component `u_m24`, strongest gust from
  225–315° `gust_w_max24`; since version 2026-10-28); "ventilation" and precipitation between the
  origin and T (`vent_o_T`, `rain_o_T`; in the backtest every hour o+k comes from the forecast of
  its own lead time, `lead_matched_vent`);
- CAMS (≤ 24 h), time of day, weekday.

Training uses archived weather forecasts (since 2024; reanalysis before); weights favour cold
months and IQAir hours (`_weights`). Retraining every 7 days (`forecast.load_or_train`); the
previous model is kept in `models_prev.pkl`.

**Model versions (`MODEL_VERSIONS` in `aqf/models.py`).** A new version for any change that alters
the forecast (inputs, algorithms), not for the weekly retraining. The model stores its version;
`load_or_train` retrains at once if `models.pkl` belongs to another version; every issued forecast
carries `model_version`. `live.versions()` gives the first and last morning forecast of each version,
`live.score()` the version of each forecast and the error by version. The site shows what changed
and from which forecast (texts `ver_what`/`ver_short` in `I18N`). Before a version switch the
previous models and backtest are kept as dated copies.

**90 days (`LongRange`, `compose_daily`).** Days 1–7 are the hourly model's daily means; after that
the norm for the date plus the day-7 anomaly decaying with τ = 4 days. Chance of a red (> 150) and
purple (> 200) day: the first week from the spread of model errors, then a smooth transition to the
share of such days on those dates in past years (±15 days).

**Export for publication (`aqf/export.py`).** The working `models.pkl` (a pickle that also holds
per-season sums of the norms and the CAMS series) is only read; a copy is written to a new folder
in `output/model_export/` and verified: models rebuilt from the published files alone give the
same forecast to the last digit on the inputs of the latest issued forecast. Each export is written
to `<name>.partial` and renamed only after all checks. The leak check runs at every export (`--check
<folder>` repeats it before an upload): 1) only the expected files with a fixed structure (norms
of the given shapes without a season axis, short number lists in config/intervals, no pickle, the
same features in the models as in config); 2) no file (JSON arrays by rows and columns, tree
thresholds and leaves, numbers in texts including table columns) contains a run of 8 consecutive
values of a source series (12 when rounded to integers, 10 to tenths): measured PM2.5, log PM2.5
and AQI by hour and day, CAMS, the weather archive, airport wind; 3) no local paths, e-mail
addresses or tokens. A self-test plants leaks in a copy of the export (measurements in a norm
table, a per-season table, daily AQI in the card, measurements in tree leaves, CAMS in config, a
pickle file); the check must catch each one or the export fails.

**80% ranges**: quantiles of backtest errors by horizon (`report.py` → `intervals.json`), checked
monthly against real errors (`monthly.check_bands`).

## 6. Checks and analyses

| Module | What it does | Where it is shown |
|---|---|---|
| `backtest.py` | forecast every 3 h through winters 2020–2026, monthly retraining, actual and archived forecast weather | `bt_hourly.parquet` |
| `report.py` | accuracy tables, 80% ranges, the site summary | `backtest_summary.json`, `intervals.json`, model-accuracy tab |
| `leakage_check.py` | data after the origin → noise, and weather forecasts issued after the origin → noise; the model and the forecast must not change | console |
| `live.py` | issued forecasts against measurements; drift monitor (`drift`) | forecast-vs-fact tab, warnings |
| `attribution.py` | a day's error: weather / pollution / unknown / recomputation | forecast-vs-fact tab |
| `explain.py` | SHAP contributions of input groups in AQI points for each day of the week | week tab, about tab |
| `stagnation.py` | past winters: errors from stagnation or from the pollution level | about tab |
| `silam.py` | copies of SILAM runs and the comparison with our forecast of the same day on the same hours, plus corrected SILAM (`silam_score.json`) | other-forecasts tab |
| `scenario.py` | two emission-cut scenarios on the World Bank's source shares; not part of the forecast | scenario tab |
| `satellite.py` | copies of Sentinel-5P NO₂ and FIRMS fires; daily series and rank correlations with the measurements (`satellite_score.json`) | satellites tab |
| `nn.py` | neural network: backtest (`nn_test.json`); `serve` gives its forecast to `aqf.forecast` (own process, 3 copies, ~1–2 min); `shadow` adds its part to an issued forecast made without it | forecast from hour 7, other-forecasts tab |
| `blend.py` | the blend without PyTorch: from `BLEND_FROM_H` = 7 the mean of LightGBM and the network in log PM2.5 | forecast |
| `monthly.py` | backtest refresh, ranges, analyses, report | `logs/monthly_report_<YYYY-MM>.md` |
| `smoke.py` | checks before and after a deploy | logs |

## 7. Site

- `dashboard.py` gathers the data (forecast, history, backtest, real use, analyses) and inserts
  their JSON into `dashboard_template.html` (every "<" escaped). The page is one file; the script
  builds three sections: Forecast (24 hours, week, 3 months), History (by year, forecast vs fact
  with the error breakdown) and Research (model accuracy, other forecasts, scenario, satellites,
  about); tab addresses `#day`, `#week`, `#season`, `#history`, `#live`, `#accuracy`, `#compare`,
  `#scenario`, `#sat`, `#about`. Four languages, SVG charts; text is inserted as text only, never
  with `innerHTML`.
- `site.py` builds the Vercel version: `index.html` (the same page + Vercel Web Analytics, no
  cookies), `api/latest.js` (the live value from `LATEST_AQI_URL`; 5 min edge cache, 30 s for the
  refresh button, 30 s in memory), `vercel.json` (security headers: CSP with hashes of the inline
  scripts, nosniff, X-Frame-Options, Referrer-Policy, Permissions-Policy).
- `output/dashboard.html` alone has no live value (it shows the value as of the forecast run); the
  author also keeps it as a claude.ai artifact, where requests to other hosts are blocked.
- When the forecast is more than 30 h old, the page says so.

## 8. Automation

- `run_daily.sh`: `aqf.forecast` (data → models → LightGBM → the network as its own process and the
  blend from hour 7; on failure LightGBM alone with the previous version and a warning → write-once
  log → error breakdown) → `aqf.silam fetch` → `aqf.satellite fetch` → `aqf.nn shadow` (adds the
  network's part if the forecast went out without it) → `aqf.dashboard` → `deploy_site.sh`. Log:
  `logs/daily.log`, ending with `ok`. A failure of SILAM, the satellites or the shadow network is
  logged as a `WARNING` line and does not stop the deploy.
- `deploy_site.sh`: `aqf.site` → `aqf.smoke build` (FAIL stops; the site is not changed) →
  `vercel deploy --prod` → `aqf.smoke live`.
- `run_monthly.sh`: `aqf.monthly` (backtest, report, ranges, error breakdown, stagnation) →
  `deploy_site.sh`.
- A local scheduler runs `run_daily.sh` every morning at about 07:15 Tashkent time (later runs the
  same day only catch up when the morning one was missed) and `run_monthly.sh` on the 1st at 09:00.

## 9. Setting up from scratch

```bash
git clone <this repository> air_quality_forecast && cd air_quality_forecast
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# LightGBM and OpenMP on macOS (repeat after reinstalling lightgbm):
install_name_tool -add_rpath @loader_path/../../sklearn/.dylibs .venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib
codesign --force -s - .venv/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.dylib
mkdir -p data/raw    # then put the target series into data/raw/tashkentairdata_air.json
.venv/bin/python -m aqf.bootstrap                  # all other data, ~30–40 min
.venv/bin/python -m aqf.backtest && .venv/bin/python -m aqf.report   # accuracy and ranges
.venv/bin/python -m aqf.attribution && .venv/bin/python -m aqf.stagnation   # analyses for the site
.venv/bin/python -m aqf.forecast --retrain         # models + first forecast
.venv/bin/python -m aqf.dashboard
```

Vercel (the account owner logs in once):

```bash
npm i -g vercel            # or locally in tools/
vercel login
.venv/bin/python -m aqf.site
cd output/site && vercel link --project <your project> && vercel git disconnect && cd ../..
printf '.env*\n.vercel\n.gitignore\n' > output/site/.vercelignore
./deploy_site.sh
```

Set `LATEST_AQI_URL` in the Vercel project for the live value (a JSON with
`{"latest": {"aqi": …, "recorded_at": …}}`); without it the function answers 502 and the page shows
the value of the forecast run. `vercel git disconnect` is required: a project connected to a
repository publishes the code at the public address on every push. The project name and the site
address are constants in `aqf/site.py`, `aqf/smoke.py` and `deploy_site.sh`.

Neural network (`aqf/nn.py`; without PyTorch the forecast goes out from LightGBM alone with the
previous version and a warning): `uv pip install --python .venv/bin/python -r requirements-nn.txt`.

## 10. Reliability and security

See README, "Reliability and security". In short: a bad forecast is not deployed (smoke tests), a
failed run leaves yesterday's site online, files are written atomically, the target cache is
protected from truncated replies, a corrupted model falls back to the previous one; the site has a
hash-based CSP, the function is protected from cache-busting, and code and secrets are not deployed.
