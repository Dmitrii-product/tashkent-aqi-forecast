"""Hourly US AQI for Tashkent: the city mean of IQAir sensors, the forecast target.

The author's copy reads this series from tashkentairdata.uz, which keeps an hourly archive of the
IQAir city data. Neither the measurements nor a downloader are part of this repository: put the
series you are allowed to use into data/raw/tashkentairdata_air.json yourself, as

    {"start": "2025-01-05T01:00:00+05:00", "values": [87, 92, null, ...]}

`start` is the first hour (ISO 8601 with its UTC offset); `values` are consecutive hourly US AQI
values (null for a missing hour), each hour labelled by its end, as on tashkentairdata.uz.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def fetch(path: Path | None = None) -> dict:
    """Check and return the locally supplied series; this copy downloads nothing."""
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"no measurements at {path}: they are not distributed with this code; "
                                "see aqf/sources/tashkentairdata.py for the expected file")
    js = json.loads(Path(path).read_text())
    if not isinstance(js, dict) or "start" not in js or not isinstance(js.get("values"), list):
        raise ValueError("unexpected format: expected {'start': ..., 'values': [...]}")
    return js


def to_series(js: dict) -> pd.Series:
    """Hourly AQI indexed by naive Tashkent local time."""
    start = pd.Timestamp(js["start"]).tz_convert("Asia/Tashkent").tz_localize(None)
    idx = pd.date_range(start, periods=len(js["values"]), freq="h")
    s = pd.Series(js["values"], index=idx, dtype="float64", name="aqi")
    return s


def load(path: Path) -> pd.Series:
    return to_series(json.loads(path.read_text()))
