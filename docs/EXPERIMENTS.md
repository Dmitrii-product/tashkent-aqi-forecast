# Catalogue of tested ideas

Everything tested, with the result and the decision. The metric is the mean absolute error in US
AQI points (MAE) over winters (October–March) unless stated otherwise. The honest test is always
the same: the model learns only from data before the forecast origin and gets the archived weather
forecast as input, as in operation (`aqf/backtest.py`). Only what is better on winters not used for
the choice was accepted (usually 2025/26 and October 2026; the choice is made on 2024/25).

## In operation

| What | How it was tested | Result |
|---|---|---|
| A separate booster for each of the 14 horizon groups | against one model with the horizon as a feature | one model lost to persistence at 1–3 h; separate ones did not |
| Merging the embassy monitor and IQAir: +1 h, scale, IQAir daily profile | correlation on the common stretch, backtest | daily values r = 0.89; without the profile the model learned the embassy's morning peak |
| Climatology without the row's own season in training | `aqf.leakage_check` | otherwise a leak: a training row's "norm" is a smoothed copy of its own target |
| Training on archived weather forecasts (since 2024) | against training on reanalysis | the model learns the same weather-forecast biases it sees in operation |
| Weather = mean of 4 models (best match, GFS, ICON, GEM) | wind forecast error, backtest | wind 3–7 days ahead −10…13%, week ≈ −1 AQI |
| Inversion features (T at 925/850/700 hPa − T at 2 m) | backtest | in every variant |
| Choice of variant by horizon (`aqf.select_horizon`) | chosen on 2024/25, tested on 2025/26 and October 2026 | L1 for 1–3 and 97–120 h, CAMS for 7–24 h, larger LightGBM elsewhere |
| Evolutionary hyper-parameter search ((μ+λ), `aqf.tune`) | 49 candidates per horizon, searched on 2024/25 | kept only for 4–6 and 37–48 h, where it was better on both unseen periods; elsewhere it overfitted |
| 90 days: norm + decaying day-7 anomaly (τ = 4) | against daily boosting | after 7 days boosting is worse than the norm |
| Chance of a red day = share of red days on those dates in past years | calibration | November stopped being "all orange" |
| Hourly model overall | winter 2025/26 | 24 hours ±26.0 → ±24.7 AQI, week ±34.2 → ±31.5 (first model → final) |
| Westerly flow over the 24 h before hour T: mean westerly surface component, strongest gust from the west (225–315°) | chosen on 2024/25 head-to-head, then the full protocol on all three periods | 1–168 h: −0.39 / −0.57 / −0.23 AQI (2024/25, 2025/26, October 2026); model version 2026-10-28 |
| Blend of LightGBM and the neural network from hour 7 (`aqf/nn.py`, `aqf/blend.py`; shadow from 2026-10-27, served from 2026-11-04, version `2026-11-04`) | network backtest with the `aqf.backtest` protocol (monthly retraining, archived weather) against LightGBM 2026-10-28 on the same pairs; then 14 forecasts in the shadow | 1–168 h: −0.46 (−0.78…−0.17) / −1.03 (−1.44…−0.63) / −2.89 (−3.90…−1.78) AQI (2024/25, 2025/26, October 2026); 1–24 h −0.1…−1.8; 1–6 h stay LightGBM. Shadow: week 30.1 → 27.8, 3 days ahead 28.1 → 22.0; in the stagnant week of 28.10–2.11 the 3-day bias −20 → −11 |

## Rejected

| Idea | Result |
|---|---|
| Weighting recent years (half-life 120/365 days), only the last 2 years, only IQAir data | worse by 1–5 AQI |
| Other algorithms: XGBoost, HistGradientBoosting, Ridge, Extra Trees, MLP | XGBoost ≈ HGB ≈ LightGBM; Ridge/ET/MLP worse by 1–6 AQI |
| Ensembles and stacking of boosting variants | −0.4 AQI for 24 hours, not adopted |
| Blending with climatology | worse |
| Baseline "last year + current trend" | 41–57 AQI for 24 hours, worse than climatology |
| CAMS inputs on all horizons | ±0.5 AQI; kept only for 7–24 h, where they won the choice |
| L1/Huber everywhere, outlier cleaning of the target (Hampel), more training rows | ±0.3, within noise |
| Quantile calibration of the forecast weather to ERA5 | over-corrects: the bias moves with Open-Meteo model updates |
| A month's anomaly carried to the same month next year | correlation −0.05 |
| The anomaly of the last 12 months for the winter | worse on winters 2023–2025 |
| Holidays and weekends (`aqf/calendar.py`) | not chosen on any horizon, effect ±3–4% |
| Correction for recent bias (`research/bias_correction_test.py`): window 7/14/28 days, overall or by lead day, weight 0.25–1 | 24 variants, none better on both winters and October 2026 at once; up to +6 AQI worse |
| Heating-season feature (`aqf/heating.py`): official dates 2019, 2021–2025, gaps filled by "5 days below +8 °C" shifted by −19 days | −0.6…+1.0 AQI; within run-to-run spread (±0.3–0.4) on repeated runs |
| Boundary-layer height in the hourly model (`research/blh_input_test.py`, 2026-10-30): BLH (log at the target hour, 24 h mean of the log, 24 h max), "ventilation" (BLH × surface wind, at the hour and over 24 h), both sets. There is no archive of BLH *forecasts* (Open-Meteo previous runs: none; near-analysis only since 2024-09), so the models got ERA5 at every lead: perfect knowledge, an upper bound. All 14 groups, 2 seeds, chosen on 2024/25 | "Both sets" chosen: −0.49 (−0.72…−0.28) on 2024/25, −0.33 (−0.54…−0.13) on 2025/26, −0.18 (−0.90…+0.55) on October 2026 (to 18.10). The whole gain is at 3–7 days (−0.80 / −0.50), where the test hands the model the *future* BLH (perfect weather as a whole gives −3.0 / −1.5 there); at 1–24 h, where the BLH forecast is close to the truth, zero (+0.05 / +0.14). The shortfall on hours with AQI > 150 barely changes. A BLH forecast is computed from the same forecast fields the model already gets, so in operation the gain ≈ 0: postponed |
| Recent wind-forecast error as an input (`research/wind_bias_test.py`, 2026-11-04): mean forecast − ERA5 surface wind over the 3 days before the issue for 1-day forecasts; the same for the matching lead; + the difference in calm hours | "1 day" chosen but worse even on the choice winter: +0.21 (−0.00…+0.41); 2025/26 +0.38, October 2026 +0.08; the shortfall at AQI > 150 grows: rejected. The error persists, but the model already sees its trace in the recent measurements |
| Wind, visibility and calm at Tashkent airport (METAR UZTT via Iowa Environmental Mesonet, since 2018) as inputs (`research/wind_bias_test.py`, 2026-11-04): over the day before the issue; forecast − airport difference over 3 days; both | "Previous day" chosen: 2024/25 −0.04 (−0.14…+0.07), 2025/26 +0.18 (+0.07…+0.28), October 2026 −0.12: rejected; "both" better only on October 2026 (−0.29) |
| Less trust in far-lead weather in LightGBM training (`research/forecast_trust_test.py`, 2026-11-05): (1) wind and rain between the issue and hour T from the forecast of the matching lead instead of the 1-day one; (2) before the forecast archive (pre-2024), the real forecast error of the same lead added to ERA5, in 7-day blocks from days before the tested winter; (3) the same for wind only. All models scored on honest rows | (1) chosen, but no better even on 2024/25 (+0.05), 2025/26 +0.23: rejected. (2): +0.13 / +0.68 / −0.91 (−1.33…−0.49), best on October 2026 but unreliable; (3): +0.39 / −0.01 / −0.48. The old computation flattered the backtest only a little (−0.13 / −0.06 / −0.29 AQI over the week); since 2026-11-05 the backtest uses the matching lead for ventilation (`lead_matched_vent`), training is unchanged |
| The network's share in the blend by horizon instead of 50/50 from hour 7 (`research/blend_weight_test.py`, 2026-11-05): one share for all hours from 7, or one per horizon group, grid 0…1 by 0.05, on the finished backtests of both models | shares chosen on 2024/25: 0 at 1–3 h, 0.1 at 4–6, 0.2–0.45 beyond; 2024/25 −0.07 (−0.17…+0.02), 2025/26 +0.05, October 2026 +0.54 (+0.28…+0.77): rejected. The best share grows with time (2024/25 ~0.4, 2025/26 ~0.5, October 2026 0.55–1.0): the network is catching up, 50/50 stays; revisit after the winter |
| Pressure around Tashkent as an input (`research/pressure_gradient_test.py`, 2026-11-04): sea-level pressure at 4 points 400–500 km away (Kazakh steppe, Kyzylorda, Bukhara, Termez) minus Tashkent's; + their 24 h change; geostrophic wind of the plane through 5 points (u, v, speed, 24 h mean speed). Mean of the 4 archived forecast models since 2024, ERA5 before | geostrophic wind chosen: 2024/25 −0.20 (−0.32…−0.07), 2025/26 −0.11 (−0.24…+0.03), October 2026 +0.42 (+0.13…+0.71): rejected; pressure differences +0.02…+0.11 on 2024/25 |
| Spread between weather models as an input (`research/weather_ensemble_test.py`, 2026-11-04): standard deviation of wind between best match, GFS, ICON, GEM at the hour and over 24 h; + temperature, pressure, 24 h precipitation; both | "wind" chosen but worse even on the choice winter: +0.14 (−0.09…+0.37); 2025/26 +0.07, October 2026 +0.03; other sets +0.3 on 2024/25: rejected |
| More models in the mean weather forecast (`research/weather_ensemble_test.py`, 2026-11-04): + ECMWF IFS 0.25° (archive from 2024-02-04, no gusts or snow), + ECMWF AIFS (AI model, archive only from 2025-02-18), both | both chosen: 2024/25 −0.33 (−0.51…−0.15), but 2025/26 +0.28 (+0.03…+0.53), October 2026 +0.26: rejected. Best match here is already almost IFS (temperature r = 0.98), so the mean would lean on one model |
| Carrying over yesterday's miss (`research/episode_checks.py`, 2026-11-04): the day's forecast shifted by φ × the log error of our previous 1-day forecast (pollution beyond the weather persists ~1 day), φ 0.1–0.5, chosen on 2024/25 | on 2024/25 φ = 0 is best: rejected by the rule; on 2025/26 −0.2, on October 2026 (19 days) −2 |
| "Same as yesterday" when the model expects a drop in the middle of an episode (`research/episode_checks.py`, 2026-11-04) | 2024–2026, AQI > 150 the day before the issue and a forecast 30+ lower (44 days): model error 27 / 24 (1 / 3 days), "same as yesterday" 67 / 69, half and half 43 / 44: the drop usually comes. A promised front does not overstate the clearing on average (weather part −2…−5 AQI, total model bias −2…+1). The week of 28.10–2.11 (weather part −15 / −33 every day) was the exception |
| Sentinel-5P NO₂ in the hourly model (`research/no2_input_test.py`, 2026-10-27): the last cloud-free overpass before the forecast day and its age; + 7-day mean; + the same for the 30–50 km ring. All 14 groups with their settings, 2 seeds, chosen on 2024/25 | all three variants worse on the choice winter (+0.09…+0.30 AQI over 1–168 h): rejected by the rule fixed in advance. For completeness: 2025/26 −0.18…−0.23, October 2026 −0.11…−0.34 at errors of 26–32 (< 1%). The model already sees the current pollution in the measurements; can be rechecked on winter 2026/27 |
| Error breakdown "over several days in a row" instead of the midnight control run | the residual with actual weather loses its link after ~1 day (autocorrelation 0.29 → 0.05) |

## Analyses (understanding, not the model)

| Question | Module | Answer |
|---|---|---|
| What a day's error is made of | `aqf/attribution.py` | 1 day ahead (backtest, ±21 AQI): weather ±9 (25%), pollution above/below expectations ±12 (32%), unknown ±15 (43%); 3 days ahead 26/37/37% |
| What share of the error comes from weather-forecast errors (2026-11-06) | `research/weather_error_share.py` | Backtest 2024-02…2026-11 against a model trained on and given the true weather. The site's forecast (issued 06:00) 1 day ahead: ±20.2 → ±18.7 with reanalysis (−8%, interval 0.6…2.5) → ±17.6 with the wind measured at the airport (−13%); 3 days ahead ±23.4 → ±20.4 (−12%) → ±18.8 (−19%). By lead of the daily product: the issue day (1–23 h) 0% / 2%, 2–3 days 9% / 14%, 4–7 days 16% / 21%. The stagnant spell of 28.10–3.11.2026: 1 day −27%, 3 days −37% (with airport temperature and humidity −49%). Airport temperature and humidity add nothing to the wind on average. The weather moves the forecast by 5/10/13 AQI on average, but on ordinary days the shift is almost as often away from the fact as towards it; the "weather" column of the error breakdown is a shift, not a share. October 2026: with perfect weather the forecast is still too high (+6…+12): the drift is not from the weather |
| Where to get a more accurate wind forecast (2026-11-06) | `research/wind_model_skill.py` | Archived Open-Meteo forecasts against the airport wind, October–March 2024–2026: our 4-model mean r 0.78 / 0.75 (1 / 3 days), reanalysis 0.72; ECMWF IFS 9 km (the "best match" member of our mean) on the same days 0.80 / 0.75 against 0.76 / 0.73 for the mean; to be retested with more weight on IFS once its 3+ day archive (from 2025-10) covers a winter; UKMO 10 km 0.73 / 0.65, AIFS 0.64 / 0.63, others lower. Global models will not do better: the gap to the measurement is the local valley wind (1–10 km scale), which needs a local high-resolution model (WRF) or national weather-service data. The Tashkent radiosonde (WMO 38457) is in IGRA only until 2001 |
| Why the forecast is what it is | `aqf/explain.py` | exact TreeSHAP; groups: recent pollution, seasonal norm, temperature, stagnation, precipitation and humidity, other |
| What pushed up the too-high forecasts of late October 2026 | `aqf/explain.py` | stagnation +22…+40 AQI, humidity +8…+12, temperature (heating) +2…+4 |
| Does the model overrate stagnation | `aqf/stagnation.py` | 1006 days, 7 winters: no (slope −0.04, r −0.02; with a monthly level −0.19 ± 0.13); monthly misses are level shifts |
| When to warn about a bias | `aqf/live.py` (`drift`) | 14 days, ≥ 7 with measurements, ≥ 15 AQI, ≥ 75% in one direction: 12% of backtest days, ~4 episodes per winter |
| Why the forecast missed 28.10–3.11.2026 | `research/late_october_miss.py` | The weather forecast promised more wind every time (14 cases: days × 1 and 3 days ahead): +1.3 / +1.6 km/h (usually 0 ± 1.4), 6 / 10 fewer calm hours a day. The stagnation was real: airport wind (METAR UZTT) 7.4 km/h against the usual 9.0. Shortfall in daytime (−13…−31 AQI at 9–17 h, usually −2…−7): the model expected daytime ventilation. AQI 160 was the maximum for these dates in 2019–2026; heating had not started, CAMS dust ordinary, fires and NO₂ high but as high on normal days. The wind-forecast error persists: autocorrelation 0.4, the error known at issue time is linked to the future one (ρ 0.22–0.29) |
| Did the airport's own forecast (TAF UZTT) see the calm of 28.10–3.11 | `research/taf_check.py` | No open archive (Iowa Mesonet: US only, Ogimet empty); aviationweather.gov keeps 30 days: 109 TAFs for 6.10–4.11. A 24 h TAF available at our 06:00 issue covers only the first ~23 hours. Its wind is a template: only 8 or 10 knots all month (15–18.5 km/h, 10 km/h above the measured), unrelated to the airport wind day to day (r 0.05; the 4-model mean 0.72). Smoke/haze in the TAF follows the AQI (r 0.3–0.46): what the forecaster sees at issue time, which the model already knows from the measurements. Not collected |
| Does the 1- and 3-day forecast catch peaks, and what causes them (2026-10-30) | `research/peaks_study.py` | below, "Pollution peaks" |

## Pollution peaks: causes and how the forecast catches them (`research/peaks_study.py`, 2026-10-30)

The site's forecast reconstructed from the backtest of version 2026-10-28: issued at 06:00, daily
mean AQI 1 and 3 days after the issue, October–March. Archived forecast weather for 2024/25, 2025/26
and October 2026; actual weather for all 7 winters 2020–2026 (an upper bound). A "sharp rise" is a
daily AQI 30+ above the previous day.

| What | 1 day ahead | 3 days ahead |
|---|---|---|
| Error after a rise of 50+ in a day (archived weather) | −35 (fact 158, forecast 123) | −27 |
| … after a rise of 30–50 | −15 | −11 |
| … after a drop of 30+ | +16 | +17 |
| Mean error of all days / "same as yesterday" | ±19.6 / ±36.6 | ±22.1 / ±42.1 |
| Slope of fact on forecast (log) | 1.01 | 1.02 |
| Red days (> 150) caught by a number > 150 / false alarms | 64% / 27% | 69% / 33% |
| … by a red-day chance ≥ 30% / share of such days that were red | 88% / 54% | 89% / 48% |
| Share of the red-day shortfall due to the weather forecast (same days, archived vs actual weather) | 19% | 28% |

- The forecast is calibrated: when it promises 160+, that is what happens on average (fact 184). It
  is not "flattened"; it does not see part of the rises in advance. Stretching it would not help.
- Missed red days are starts of episodes in moderate weather: wind 4.5 against 3.7 km/h for caught
  ones, 15 against 20 calm hours, night inversion 1.8 against 4.6 °C, the previous day's level 109
  against 159; October and January–February. Caught ones are the deep stagnation of
  November–December.
- What the model underestimates with correct weather is linked to almost nothing measurable
  (|ρ| ≤ 0.13: wind, calm, inversion at 850/925, humidity, snow, NO₂, weekday). Only to yesterday's
  miss (ρ 0.32: extra pollution persists about a day) and to dry spells longer than 14 days
  (shortfall 12, 41 days; "days without rain" as an input was tested and not chosen).
- Causes of peaks: 64% of red days are deep stagnation (16+ calm hours or a night inversion of
  4 °C and more; only 31% of cold-season days are like that), 54% are cold (< 3 °C, heating). With
  both conditions a day is red in 54% of cases, with neither in 8%. 187 of 226 red days fall in
  November–January. On days of a sharp rise the wind is weaker (−0.5 σ from the month's norm), calm
  longer (+0.5 σ), the wind dropping from the day before, pressure higher, colder, no rain.
- Hourly peaks: on days whose dirtiest hour was above 200 (79 days), the forecast for it was 189
  against a fact of 256.
- A next-day forecast issued later (same days, archived weather): from 06:00 ±19.6, 12:00 ±18.8,
  18:00 ±17.4, 21:00 ±16.0; red days caught 64 → 72%, but after a rise of 50+ still −30 (instead of
  −35): sharp rises come without warning even in the evening.

Conclusion: the underestimation of peaks is mainly pollution beyond what the weather explains, not
a weather-forecast error and not a "flattened" model. The best peak signal already exists: the
red-day chance.

## Neural network next to LightGBM (`aqf/nn.py`, 2026-10-27)

A GRU reads the last 72 h of pollution and actual weather; a shared decoder for each of the 168 hours
ahead takes the weather forecast for that hour (by lead time), calendar, the norm (without the
row's own season in training), ventilation and precipitation since the origin. Protocol as in the
backtest: monthly retraining, a forecast every 3 h, 3 seeds (their forecasts averaged), 469 thousand
(origin, hour) pairs.

| | 24 h: LightGBM | network | blend 50/50 | week: LightGBM | network | blend |
|---|---|---|---|---|---|---|
| winter 2024/25 | 26.5 | 28.1 | 26.5 | 32.5 | 33.1 | 32.0 |
| winter 2025/26 | 24.7 | 25.5 | 24.2 | 31.5 | 31.0 | 30.2 |
| October 2026 | 22.2 | 20.4 | 20.3 | 27.0 | 24.1 | 24.5 |

The network alone is worse at 1–24 h. The blend (mean in log PM2.5) beats LightGBM on all horizons
from 7 h in all three periods (−0.1…−1.0, −0.5…−1.8, −0.3…−4.6 AQI by group); at 1–3 h it is worse in
all three (+0.2…+0.8), at 4–6 h mixed (+0.5, −0.2, −1.2). The 50/50 weight was set before the test;
the "from 7 h" threshold was chosen from its results. The blend from hour 7 (LightGBM at 1–6 h): 24 h
26.3 / 24.2 / 20.4, week 31.9 / 30.2 / 24.6 (column `blend7` in `nn_test.json`).

**Shadow from 2026-10-27.** Every morning `aqf.nn shadow` trained three copies on all data up to the
issued forecast's origin, with its own weather, and wrote the network's and the blend's forecast;
`aqf.live.shadow` compared the three on the same hours and days. Served since 2026-11-04 (see "In
operation").

## The ecology committee's statement on drought and dust (2026-10-27)

| Question | How it was checked | Result |
|---|---|---|
| Does the model err in long dry spells | next-day error with actual weather (1006 winter days) against days since the last rain ≥ 1 mm (ERA5) | in the driest fifth of days (6–39 days without rain) the model underestimates by 10 AQI on average, elsewhere −4 to 0; within a month −0.9 ± 0.4 AQI per dry day (r −0.15). The model remembers precipitation for 72 h only. Tested as an input on 2026-10-28 (`research/westerly_wind_test.py`, variant "dry"): −0.02 on the choice winter (noise), −0.24 on 2025/26, −0.10 in October 2026: not chosen |
| October 2026 precipitation | ERA5 via Open-Meteo against the committee's figure | ERA5 21 mm for 1–26.10, the committee's stations 3.9 mm: the reanalysis counts drizzle. The main point agrees: no rain since 12.10. Station precipitation for Tashkent is not openly available (NOAA GHCN stops in 2025-08, no GSOD for 2026) |
| Did CAMS see the dust of 25–26.10 | CAMS dust by day | 0.3–0.8 µg/m³ against a winter median of 4.6: CAMS missed the cloud seen on Meteosat images. Over 4 winters CAMS dust is unrelated to our errors (r −0.07) |
| SILAM as a model input | FMI open data | kept ~32 days, no history to train on; copied daily since 2026-10-27 (`aqf/silam.py`) and compared with our forecast |
| Is SILAM more accurate than our forecast | 32 SILAM runs (26.09–27.10) against 10 of our forecasts and the measurements on the same hours; SILAM of the same day as ours (05:00 against our 06:00), as is (`aqf.silam score`) | each hour of the first 24: ours ±20.1 AQI, SILAM ±53.9 (216 h); days 2–3 ±25.8 against ±52.9 (359 h); next-day daily mean ±20.9 against ±52.1 (8 days). SILAM underestimates by ~50 AQI (its PM2.5 about a third of the measured) and catches rises and falls worse (r 0.27–0.63 by hour and day against 0.7–0.97 for ours). The freshness of the SILAM run barely matters. Our forecast was too high in those weeks (+11…+27) and still more accurate. Few days for a final verdict; the comparison continues |
| Why SILAM underestimates: does it miss the current pollution? | `research/silam_bias_check.py`: 32 runs against measurements, by lead, by hour of day, by week; SILAM corrected by recent measurements | It starts without the real pollution (SILAM does not assimilate measurements) but that is not why it is low: measured/SILAM 2.3 in the first 6 h and 2.0–2.4 up to 5 days; no night peak (night/day 0.88 against 1.47 measured); the gap grows when heating starts (3.3 from 26.10). City sources are missing. Correction by the ratio of the last 7 days: next day ±48 → ±21 AQI. CAMS is also 2.3 times below the measurements in winter. Corrected SILAM is tracked next to the original since 2026-10-27 |
| Who was right about 28.10–2.11: the committee ("conditions will persist until the start of next week") or our forecast (a respite on 29.10, a drop from 1.11) | rules fixed in advance on 2026-10-27, judged on 2026-11-04 | Persistence: both right (every day above 100). Shape of the week: the committee was right, there was neither a respite on 29.10 nor a drop on 1–2.11, and the air stayed polluted without rain. Numbers: ours were far closer (mean error 29 AQI for 28–31.10 against 85 for SILAM's run of 27.10), but we underestimated all six days (mean −34) |

## Satellites: Sentinel-5P NO₂ and FIRMS fires (`aqf/satellite.py`, 2026-10-27)

Observations next to the measurements, not model inputs. Rank correlation (Spearman) with the daily
AQI.

| What | Days | Result |
|---|---|---|
| NO₂ over the city (≤ 15 km, qa ≥ 75), 2018-06 – 2026-10 | 2042 with measurements | all days 0.44; winter (Nov–Feb) 0.52 (550); September–October 0.29; last 90 days 0.49. With the AQI at 13–14 h (overpass time) no better: 0.40 / 0.44 |
| NO₂ in the 30–50 km ring | 2269 | 0.47 / 0.47 / 0.37: the regional background is as related to the fact as the city's |
| VIIRS fires within 200 km (without industrial heat sources) | 2305 | no relation: all days −0.13 (fires come in summer, when the air is cleaner), winter 0.04, September–October 0.14; with the next day's AQI 0.05–0.16 |

NO₂ as a model input was tested on 2026-10-27 and rejected (see "Rejected"). The fast NRTI product
is in the same MEEO copy and since December 2025 arrives 1.5–3 h after the overpass: yesterday's
would be available for a 06:00 forecast.

## Westerly wind and dust: an IQAir note of 2026-10-26 (2026-10-28)

IQAir wrote that dust drove the AQI above 150 on 26.10: an "acute autumn dust episode", PM10 as the
main pollutant, dust from the Kyzylkum and Kazakhstan brought by strong westerly winds, and low NO₂,
CO, SO₂ and O₃ showing that traffic and industry were not the cause. Checked here, together with
whether westerly wind is useful as a model input.

**Westerly wind as an input** (`research/westerly_wind_test.py`, then `research/westerly_wind_full.py`).
The model already sees the wind at every hour (u, v at the surface and at 850 hPa, gusts, 24 h mean
speed) but not how much and how hard it blew from the west before that hour.

| What | 2024/25 (choice) | 2025/26 | October 2026 |
|---|---|---|---|
| "west": mean westerly component over 24 h at the surface and at 850 hPa, strongest gust from the west (225–315°) over 24 h; head-to-head, chosen | −0.49 | −0.75 | −0.19 (at 73–168 h +0.49) |
| the same without the 850 hPa wind (in the history it comes from analyses, as there is no archive of upper-air forecasts, so it may know too much); tested afterwards as a correction, not a choice | −0.42 | −0.67 | −0.56 |
| full protocol (`aqf/backtest.py`: monthly retraining, model choice by horizon), surface variant against the current model | **−0.39** [−0.57…−0.22] | **−0.57** [−0.74…−0.40] | **−0.23** [−0.69…+0.20] |

The current model's error in the full protocol is ±32.5 / ±31.5 / ±26.2 (1–168 h), so the gain is
1–2%; at 1–24 h −0.15 / −0.46 / −0.30, at 25–72 h −0.53 / −0.67 / −0.74. The leakage check with the
new inputs is clean. Better on both periods not used for the choice; in October 2026 (27 days) the
interval includes zero. **Adopted on 2026-10-28** as model version `2026-10-28`. The monthly
backtest recomputed for the new version (winters 2020–2026, reanalysis weather for 2020–2023) is
better on all seven winters. The meaning, judging by the data, is not dust: in October–March days
with westerly gusts above 30 km/h are cleaner (AQI 88–93 against 125 with gusts up to 20 km/h): a
westerly flow in Tashkent means a front and ventilation.

**Dust or the city** (`research/dust_claim_check.py`; the Sentinel-5P aerosol index from the same
open MEEO copy as NO₂, September–November 2018–2026; the zero is shifted in 2018–2020, the base is
2021–2025).

| Claim | What the data show |
|---|---|
| dust is 52–78% of PM2.5 (64% on average) | a figure from a ministry briefing of 21.10: laboratory analysis of samples from 28.09–6.10 (13–14.10: 47–52%), not from 24–27.10. Plausible for early October: before heating (in the World Bank's shares dust from outside and city dust are 54% in October), and 20.09–10.10.2026 was the cleanest such period in all years (AQI 78): a large share of a small amount |
| strong westerly winds brought the dust | ERA5 24–27.10: mean wind 4–5 km/h, gusts up to 23 km/h, from the west up to 23; peaks at night in calm and with air draining from the mountains in the east (night/day 1.12–1.22, within October's usual 1.09–1.44) |
| low NO₂ means not traffic or industry | satellite NO₂ over the city on 23, 24, 26.10 at the 76th–93rd percentile of Octobers; 25.10 at the median |
| a dust episode | CAMS dust ≈ 0. Our model does not see dust, yet its 24 h forecasts matched the rise: the weather (stagnation) explains it. Beyond the weather the air was dirtier than expected by 0 / 7 / 21 AQI (25 / 26 / 27.10), an upper bound of an outside contribution |
| — | the aerosol index on 25–26.10 was high for autumn (92nd–93rd percentile): absorbing aerosol (dust or smoke) was present. But it was as high on 20–21.10 with AQI 93–97, and in October it is unrelated to the daily AQI (r 0.06–0.17; NO₂ 0.48); in November it is related (0.37–0.42): heating smoke, which the index also sees |

Conclusion: the rise of 25–27.10 was above all stagnation in which city pollution accumulated; an
aerosol layer, probably with dust, was present and could have added up to ~20 AQI, but the data do
not support "dust, not the city". We have no PM10, so its dominance at 13:00 on 26.10 cannot be
checked.

## Other studies

| Work | What we did | Conclusion |
|---|---|---|
| NUU poster (Madiyarova et al., ECAS-7 2025) | reproduced on the same data (`research/repro_nuu_poster.py`) | the figures (Lasso MAE 5.5 µg/m³, R² 0.84) come out only with the previous hour's measurement among the inputs; "repeat the previous hour" gives R² 0.835: a 1-hour forecast, not a 24-hour one |
| Akhatkulov et al., ICAIIT 2026 (16 stations, LR/RF/SVR/GRU, R² 0.98) | read the abstract | horizon and split not given; R² 0.98 looks like the same neighbouring-hours effect |
| Yalgoshev, review 2026 (3 stations) | read the abstract | no reliable 24-hour model found |
