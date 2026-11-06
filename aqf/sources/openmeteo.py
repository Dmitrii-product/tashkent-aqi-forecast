"""Weather from Open-Meteo (free, no key).

- archive: ERA5-based reanalysis, hourly since 1940 — used to train the model.
- forecast: up to 16 days ahead (ECMWF IFS / best-match blend) — model input at run time.
- previous runs: archived forecasts as they were issued 1..7 days before (since ~2024) —
  lets the backtest feed the model the same imperfect weather forecast it gets in production.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

LAT, LON = 41.3111, 69.2797  # Tashkent centre
TZ = "Asia/Tashkent"

VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "snowfall",
    "cloud_cover",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
]


# Upper air: temperature above the city vs at the surface gives inversion strength, the main
# driver of winter smog in Tashkent. Archived (near-analysis) since 2021-03.
UPPER_VARS = ["temperature_925hPa", "temperature_850hPa", "temperature_700hPa",
              "wind_speed_850hPa", "wind_direction_850hPa"]


def _get(url: str, params: dict, retries: int = 4) -> dict:
    # free tier counts big requests (many variables x many days) as several calls and
    # rate-limits per minute: back off on 429 instead of failing
    for attempt in range(retries + 1):
        r = requests.get(url, params=params, timeout=120)
        if r.status_code != 429 or attempt == retries:
            break
        time.sleep(65)
    r.raise_for_status()
    js = r.json()
    if js.get("error"):
        raise RuntimeError(js.get("reason"))
    return js


def _to_frame(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def fetch_archive(start: str, end: str, lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    frames = []
    # chunk by year to keep responses small
    for y0 in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{y0}-01-01"))
        e = min(pd.Timestamp(end), pd.Timestamp(f"{y0}-12-31"))
        js = _get("https://archive-api.open-meteo.com/v1/archive", {
            "latitude": lat, "longitude": lon, "hourly": ",".join(VARS),
            "start_date": f"{s:%Y-%m-%d}", "end_date": f"{e:%Y-%m-%d}", "timezone": TZ,
        })
        frames.append(_to_frame(js))
    return pd.concat(frames).dropna(how="all")


def fetch_upper_history(start: str, end: str, lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    frames = []
    for y0 in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{y0}-01-01"))
        e = min(pd.Timestamp(end), pd.Timestamp(f"{y0}-12-31"))
        js = _get("https://historical-forecast-api.open-meteo.com/v1/forecast", {
            "latitude": lat, "longitude": lon, "hourly": ",".join(UPPER_VARS),
            "start_date": f"{s:%Y-%m-%d}", "end_date": f"{e:%Y-%m-%d}", "timezone": TZ,
        })
        frames.append(_to_frame(js))
    return pd.concat(frames).dropna(how="all")


def fetch_forecast(past_days: int = 7, forecast_days: int = 16,
                   lat: float = LAT, lon: float = LON) -> pd.DataFrame:
    js = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "hourly": ",".join(VARS + UPPER_VARS),
        "past_days": past_days, "forecast_days": forecast_days, "timezone": TZ,
    })
    return _to_frame(js)


def fetch_forecast_models(models: list[str], past_days: int = 10, forecast_days: int = 16,
                          lat: float = LAT, lon: float = LON) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Surface forecast from several global models (one frame each, same column names) and
    the upper-air forecast from the first (best-match) model."""
    js = _get("https://api.open-meteo.com/v1/forecast", {
        "latitude": lat, "longitude": lon, "hourly": ",".join(VARS + UPPER_VARS),
        "models": ",".join(models), "past_days": past_days, "forecast_days": forecast_days,
        "timezone": TZ,
    })
    df = _to_frame(js)
    frames = []
    for m in models:
        cols = {f"{v}_{m}": v for v in VARS if f"{v}_{m}" in df}
        frames.append(df[list(cols)].rename(columns=cols).reindex(columns=VARS))
    up = {f"{v}_{models[0]}": v for v in UPPER_VARS if f"{v}_{models[0]}" in df}
    return frames, df[list(up)].rename(columns=up)


def fetch_previous_runs(start: str, end: str, lead_days: range = range(1, 8),
                        lat: float = LAT, lon: float = LON, model: str | None = None) -> dict[int, pd.DataFrame]:
    """{lead_day: hourly frame} — value at time T as forecast `lead_day` days before T."""
    out: dict[int, list[pd.DataFrame]] = {d: [] for d in lead_days}
    months = pd.date_range(pd.Timestamp(start).normalize(), end, freq="MS")
    if len(months) == 0 or months[0] > pd.Timestamp(start):
        months = months.insert(0, pd.Timestamp(start).normalize())
    for i, s in enumerate(months):
        e = min(pd.Timestamp(end), (s + pd.offsets.MonthEnd(0)))
        names = [f"{v}_previous_day{d}" for d in lead_days for v in VARS]
        params = {"latitude": lat, "longitude": lon, "hourly": ",".join(names),
                  "start_date": f"{s:%Y-%m-%d}", "end_date": f"{e:%Y-%m-%d}", "timezone": TZ}
        if model:
            params["models"] = model
        js = _get("https://previous-runs-api.open-meteo.com/v1/forecast", params)
        time.sleep(2)
        df = _to_frame(js)
        for d in lead_days:
            sub = df[[f"{v}_previous_day{d}" for v in VARS]]
            sub.columns = VARS
            out[d].append(sub)
    return {d: pd.concat(v).dropna(how="all") for d, v in out.items()}


def cache_archive(path: Path, start: str, end: str) -> pd.DataFrame:
    from ..safe_io import to_parquet
    df = fetch_archive(start, end)
    to_parquet(df, path)
    return df
