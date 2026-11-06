"""CAMS global PM2.5 / dust from Open-Meteo's air-quality API (free, no key).

CAMS underestimates Tashkent's level about twofold (it misses local winter heating
emissions) but tracks the meteorology-driven ups and downs: day-to-day changes correlate
0.65 with IQAir in the cold season. Open-Meteo keeps no archive of CAMS forecasts by lead
time, only a stitched near-real-time series, so it is used for horizons up to 24 h only,
where that series matches what is available at forecast time.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from ..safe_io import to_parquet
from .openmeteo import LAT, LON, TZ

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
VARS = ["pm2_5", "pm10", "dust"]


def _frame(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").sort_index()


def fetch_history(start: str, end: str) -> pd.DataFrame:
    frames = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01"))
        e = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        r = requests.get(URL, params=dict(latitude=LAT, longitude=LON, hourly=",".join(VARS),
                                          start_date=f"{s:%Y-%m-%d}", end_date=f"{e:%Y-%m-%d}",
                                          timezone=TZ), timeout=120)
        r.raise_for_status()
        frames.append(_frame(r.json()))
    return pd.concat(frames).dropna(how="all")


def fetch_forecast(past_days: int = 3, forecast_days: int = 5) -> pd.DataFrame:
    r = requests.get(URL, params=dict(latitude=LAT, longitude=LON, hourly=",".join(VARS),
                                      past_days=past_days, forecast_days=forecast_days,
                                      timezone=TZ), timeout=120)
    r.raise_for_status()
    return _frame(r.json()).dropna(how="all")


def update_cache(path: Path) -> pd.DataFrame:
    """Append the latest stitched series (and keep collecting forecasts for later checks)."""
    old = pd.read_parquet(path) if path.exists() else None
    start = "2022-08-01" if old is None else f"{old.index.max() - pd.Timedelta(days=5):%Y-%m-%d}"
    end = f"{pd.Timestamp.now(tz=TZ).tz_localize(None):%Y-%m-%d}"
    new = fetch_history(start, end).dropna(how="all")
    if new.empty:  # an empty reply must not overwrite the last days with nothing
        raise ValueError("empty CAMS reply")
    df = new if old is None else pd.concat([old[old.index < new.index.min()], new])
    to_parquet(df, path)
    return df
