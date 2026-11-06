"""Reproduction of an external result: the NUU poster on PM2.5 prediction in Tashkent.

Madiyarova U., Erkinov J., Babaa M. R. "Particulate Matter (PM2.5) Prediction in Tashkent Using
Machine Learning". 7th International Electronic Conference on Atmospheric Sciences (ECAS-7,
MDPI), June 2025. https://sciforum.net/paper/view/23148

The poster: US Embassy hourly PM2.5, January 2023 - October 2024; temperature, humidity,
precipitation, timestamp parts and a heating-season flag (November-March); Min-Max scaling;
Linear / Ridge / Lasso / Random Forest / XGBoost. Reported (PM2.5, ug/m3):

    Linear  MAE 5.62  RMSE 10.41  R2 0.834      Random Forest  MAE 6.29  RMSE 12.75  R2 0.751
    Ridge   MAE 5.58  RMSE 10.35  R2 0.835      XGBoost        MAE 6.26  RMSE 12.54  R2 0.758
    Lasso   MAE 5.50  RMSE 10.33  R2 0.836

Same public data here (our AirNow cache and Open-Meteo archive), four set-ups: with or without
the previous hour's PM2.5 as an input, random or chronological 80/20 split. Result
(2026-10-27): only "previous hour + chronological split" reproduces the poster - linear models
MAE 5.90, RMSE 10.46, R2 0.844; Random Forest / XGBoost R2 0.75 / 0.74 - and simply repeating
the previous hour, with no model, already gives MAE 6.07, R2 0.835. Without the previous hour,
weather and calendar explain about 10% of the variance (R2 0.10). So the poster's models are
in effect 1-hour-ahead forecasts that know the current value; the high R2 comes from
hour-to-hour persistence, not from the weather.

This is research, not part of the forecast service.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        PYTHONPATH=. .venv/bin/python research/repro_nuu_poster.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from aqf.data import EMBASSY, load_weather
from aqf.sources import airnow

warnings.filterwarnings("ignore")
BASE = ["temperature_2m", "relative_humidity_2m", "precipitation", "year", "month", "day", "hour", "heating"]


def dataset() -> pd.DataFrame:
    pm = airnow.load_hourly(EMBASSY)
    pm = pm[(pm.index >= "2023-01-01") & (pm.index < "2024-11-01")]
    d = pd.DataFrame({"pm": pm}).join(load_weather()[BASE[:3]], how="left")
    d["year"], d["month"], d["day"], d["hour"] = d.index.year, d.index.month, d.index.day, d.index.hour
    d["heating"] = d["month"].isin([11, 12, 1, 2, 3]).astype(int)
    d["lag1"] = d["pm"].shift(1)  # the previous hour's PM2.5
    return d.dropna(subset=["pm", *BASE[:3]])


def main() -> pd.DataFrame:
    d = dataset()
    print(f"{len(d)} hours with PM2.5, {d.index.min():%Y-%m-%d} .. {d.index.max():%Y-%m-%d}")
    models = {"Linear": LinearRegression(), "Ridge": Ridge(alpha=1.0), "Lasso": Lasso(alpha=0.001),
              "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=4, random_state=0),
              "XGBoost": xgboost.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)}
    rows = []
    for name_f, feats in (("weather+time", BASE), ("weather+time+previous hour", BASE + ["lag1"])):
        e = d.dropna(subset=feats)
        for split in ("random 80/20", "by time 80/20"):
            te = (np.random.default_rng(0).random(len(e)) < 0.2) if split.startswith("random") \
                else np.arange(len(e)) >= int(0.8 * len(e))
            sc = MinMaxScaler().fit(e.loc[~te, feats])
            Xtr, Xte = sc.transform(e.loc[~te, feats]), sc.transform(e.loc[te, feats])
            ytr, yte = e.loc[~te, "pm"].values, e.loc[te, "pm"].values
            preds = {k: m.fit(Xtr, ytr).predict(Xte) for k, m in models.items()}
            if "lag1" in feats:
                preds["persistence (= previous hour)"] = e.loc[te, "lag1"].values
            for k, p in preds.items():
                rows.append({"features": name_f, "split": split, "model": k, "MAE": mean_absolute_error(yte, p),
                             "RMSE": mean_squared_error(yte, p) ** 0.5, "R2": r2_score(yte, p)})
    out = pd.DataFrame(rows).round(3)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    main()
