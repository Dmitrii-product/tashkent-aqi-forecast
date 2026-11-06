# research/: studies outside the service

Scripts of one-off checks. The forecast and the site do not use them, but any of them can be
repeated: each reads the same data cache (`data/`, see `python -m aqf.bootstrap`) and describes its
question, method and result (as of the check) at the top of the file.

| Script | What it checks | Result |
|---|---|---|
| `repro_nuu_poster.py` | Reproduction of the NUU poster (Madiyarova et al., ECAS-7, 2025): embassy PM2.5 2023–2024, weather + time, LR/Ridge/Lasso/RF/XGBoost | The poster's figures (Lasso MAE 5.5, R² 0.84) come out only with the previous hour's measurement among the inputs and a time-ordered split; simply repeating the previous hour gives R² 0.835. It is a 1-hour forecast, not a 24-hour one |
| `silam_bias_check.py` | Why SILAM underestimates Tashkent: error by lead time, daily cycle, the gap growing with heating, what the current measurement adds, comparison with CAMS | SILAM starts without the real pollution but that is not why it is low: 2.0–2.4 times below the measurements from the first hours to 5 days, no night peak; city sources (heating) are missing. Correction by the last 7 days of measurements: ±48 → ±21 AQI for the next day. CAMS is also low (2.3 times in winter) |
| `wb_scenario.py` | A scenario on the World Bank source shares (2024, fig. 18; the shares and computation now live in `aqf/scenario.py`): heating −30%, dust from outside −5%, transport and heavy industry −10% | PM2.5 −15…−16% in winter, −9% in October; on past winters a third fewer days with AQI > 150 (211 → 140 over 8 winters) |
| `scenario_accumulation.py` | Is the scenario's gain larger on days when pollution accumulates over several days | Accumulation alone does not change the percentage (concentration is linear in emissions). The mix does: if dust from outside stays at the month's mean and everything above it is city sources, PM2.5 drops 18% instead of 15% on days with AQI > 200 and 11% on clean days |
| `westerly_wind_test.py` | Westerly wind over the last 24 h (surface and 850 hPa, gusts from the west) and days without rain as model inputs, chosen on 2024/25 | "Westerly wind" chosen; without 850 hPa (an analysis in the history) better on all three periods: −0.42 / −0.67 / −0.56 AQI. Days without rain not chosen |
| `westerly_wind_full.py` | Surface westerly wind in the full backtest protocol (monthly retraining, choice by horizon) and the leakage check | −0.39 / −0.57 / −0.23 AQI (October 2026 within noise), no leakage; adopted on 2026-10-28 (model version `2026-10-28`) |
| `dust_claim_check.py` | An IQAir note of 2026-10-26, "dust, not the city": Sentinel-5P aerosol index, NO₂, ERA5 wind, daily cycle, error breakdown | Not supported: the rise of 25–27.10 was stagnation and city pollution (a forecast without dust caught it), NO₂ above usual, no strong wind; aerosol was present. The 64% dust share comes from samples of 28.09–6.10 |
| `peaks_study.py` | Pollution peaks: their causes and how much the 1- and 3-day forecast underestimates them (backtest of version 2026-10-28, 06:00 issue, archived and actual weather, 7 winters) | The forecast lags: after a rise of 50+ in a day it is 35 too low (27 at 3 days), after a drop of 30+ 16 too high. Calibrated (slope 1.01), so stretching will not fix it. Weather is ~20% of the shortfall at 1 day, ~30% at 3; the rest is pollution beyond the weather, persisting about a day. Red days: a number > 150 catches 64–69%, a red-day chance ≥ 30% catches 88% |
| `no2_input_test.py` | Yesterday's Sentinel-5P NO₂ in the hourly model: three feature sets, all 14 horizon groups, chosen on 2024/25, tested on 2025/26 and October 2026 | Rejected: worse on the choice winter (+0.1…+0.3 AQI), better by 0.1–0.3 elsewhere at an error of ~30 (< 1%) |
| `blh_input_test.py` | Boundary-layer height (ERA5, perfect knowledge, an upper bound) and "ventilation" in the hourly model, chosen on 2024/25 | Postponed: −0.49 / −0.33 / −0.18 AQI, but only at 3–7 days, where the test knows the future BLH; zero at 1–24 h, peaks unchanged. A real BLH forecast would add almost nothing |
| `wind_bias_test.py` | Recent wind-forecast error and airport wind/visibility (METAR) as inputs, chosen on 2024/25 | Both rejected: the wind error did not help even on the choice winter (+0.2); the airport ~0 on 2024/25 and worse on 2025/26 (+0.18) |
| `late_october_miss.py` | Why the forecast missed 28.10–3.11.2026: forecast weather against actual by variable, shortfall by hour of day, this week against the same dates of past years (AQI, NO₂, fires, dust), airport wind (METAR) | The weather forecast promised more wind every day (+1.3…1.6 km/h, 6–10 fewer calm hours), the shortfall was in daytime; the stagnation is confirmed by the airport wind. Wind-forecast errors persist for several days |
| `taf_check.py` | The airport's own forecast (TAF UZTT, 30 days from aviationweather.gov) against METAR and the 4-model mean in the week of 28.10–3.11 | TAF wind is a template of 8–10 knots, unrelated to the measurements (r 0.05 against 0.72 for the models); smoke/haze repeats the current AQI. Not usable as an input |
| `blend_weight_test.py` | The network's share in the blend by horizon instead of 50/50, on the finished backtests of both models | Rejected: a smaller share is slightly better only on 2024/25; the best share grows with time, 50/50 stays |
| `forecast_trust_test.py` | Ventilation and washout from the forecast of the matching lead (not the 1-day one) and a weather-forecast error mixed into pre-2024 training data for LightGBM | Rejected: the corrected ventilation is no better (the old one flattered the backtest by 0.1–0.3 AQI); the mixed-in error is better only on October 2026 (−0.91) and worse on 2025/26 (+0.68) |
| `weather_error_share.py` | What share of the forecast error comes from weather-forecast errors: the same backtest with archived forecast weather and with true weather (a model trained on reanalysis; and with the wind measured at the airport) | 1 day ahead ±20.2 → ±18.7 (reanalysis, −8%) → ±17.6 (airport wind, −13%); 3 days ahead ±23.4 → ±20.4 (−12%) → ±18.8 (−19%); in the stagnant spell of 28.10–3.11.2026 −27% / −37%. The rest (±17–19) is not the weather |
| `wind_model_skill.py` | Whose archived wind forecast is closest to the wind measured at the airport | Our 4-model mean: r 0.78 / 0.75 (1 / 3 days), reanalysis 0.72; the best single model is ECMWF IFS 9 km (the "best match" member of our mean), 0.80 / 0.75 on the same days; UKMO 10 km, the AI model AIFS, GFS, ICON, GEM, CMA, JMA are further. Global models will not do better; the gap is the local valley wind. The Tashkent radiosonde is archived only until 2001 |
| `pressure_gradient_test.py` | Pressure at four points 400–500 km away (Kazakhstan, Kyzylorda, Bukhara, Termez) and the geostrophic wind as inputs, chosen on 2024/25 | Rejected: geostrophic wind better on 2024/25 (−0.20), −0.11 on 2025/26 within noise, worse on October 2026 (+0.42) |
| `weather_ensemble_test.py` | Spread between weather models as an input; ECMWF IFS and AIFS in the mean weather forecast; chosen on 2024/25 | Both rejected: the spread did not help even on the choice winter; the new models helped on 2024/25 (−0.33) and hurt on 2025/26 (+0.28) |
| `episode_checks.py` | After the stagnant week of 28.10–2.11: "same as yesterday" in an episode when the model expects a drop; carrying over yesterday's miss | Both rejected: in episodes the drop usually comes (model 24–27 against 67–69 for "same as yesterday"); carrying over the miss is worse on the choice winter |
| `bias_correction_test.py` | Correcting the forecast by the mean error of the last 7/14/28 days (24 variants) | Rejected: no variant is better on both winters and October 2026 at once |

Studies that became part of the package (their results are on the site or in the README) live in
`aqf/`: `stagnation.py` (stagnation or pollution level), `heating.py` (heating-season feature,
rejected), `nn.py` (the neural network next to LightGBM), `tune.py`, `experiments.py`,
`select_horizon.py`, `leakage_check.py`. The full catalogue of checks: `docs/EXPERIMENTS.md`.

Running (from the repository root):

```bash
DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
    PYTHONPATH=. .venv/bin/python research/repro_nuu_poster.py
PYTHONPATH=. .venv/bin/python research/bias_correction_test.py
```
