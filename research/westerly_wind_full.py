"""Westerly wind under the production protocol: the full walk-forward backtest of aqf.backtest
(monthly retraining, per-horizon router, archived forecast weather) with and without the two
surface-wind inputs that passed the head-to-head (research/westerly_wind_test.py, "west_sfc"):
the 24 h mean westerly wind component at 10 m and the strongest gust from the west (225–315°)
in the last 24 h, both from the archived weather forecasts the model gets.

Both runs on the same data, winters 2024/25 (where the variants were chosen), 2025/26 and
October 2026; then the leakage check with the new inputs. Nothing in production changes: the
inputs are patched in for this run only, results go to data/processed/bt_hourly_{base,west}.parquet.

    DYLD_FALLBACK_LIBRARY_PATH=.venv/lib/python3.12/site-packages/sklearn/.dylibs \
        PYTHONPATH=. .venv/bin/python research/westerly_wind_full.py       # ~45 min

Result (2026-10-28), MAE change with the two inputs (95% interval over days), archived forecast weather:
               current MAE   1–24 h          25–72 h          73–168 h        1–168 h
  2024/25      32.5          -0.15 (n.s.)    -0.53            -0.39           -0.39 (-0.57..-0.22)
  2025/26      31.5          -0.46           -0.67            -0.55           -0.57 (-0.74..-0.40)
  Oct 2026     26.2          -0.30 (n.s.)    -0.74            +0.10 (n.s.)    -0.23 (-0.69..+0.20)
Leakage check with the inputs: forecasts identical bit for bit, the negative control caught.
Better on both periods not used for the choice; October 2026 is 27 days, its interval spans 0.
Switching the production model is the owner's call (asked 2026-10-28).
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from aqf import aqi, backtest, data, features, leakage_check
from aqf.data import PROC
from research.no2_input_test import boot
from research.westerly_wind_test import extras

SEASONS = [2024, 2025, 2026]
NEW = ["x_u24", "x_wgust24"]
_plain = features.weather_features


def with_west(w: pd.DataFrame) -> pd.DataFrame:
    return _plain(w).join(extras(w)[NEW])


def patch(on: bool) -> None:
    f = with_west if on else _plain
    features.weather_features = data.weather_features = backtest.weather_features = f


def run(name: str, on: bool) -> pd.DataFrame:
    patch(on)
    try:
        print(f"backtest: {name}", flush=True)
        df = backtest.backtest_hourly(SEASONS)
    finally:
        patch(False)
    df.to_parquet(PROC / f"bt_hourly_{name}.parquet")
    return df


def compare() -> dict:
    cols = ["origin", "h", "pred_aqi", "actual_aqi", "weather", "season"]
    a, b = (pd.read_parquet(PROC / f"bt_hourly_{n}.parquet", columns=cols) for n in ("base", "west"))
    a, b = (x[x["weather"] == "forecast"].set_index(["origin", "h"]) for x in (a, b))
    j = a.join(b[["pred_aqi"]].rename(columns={"pred_aqi": "west"}), how="inner").rename(columns={"pred_aqi": "current", "actual_aqi": "actual"}).reset_index()
    bands = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
    out = {}
    for y in SEASONS:
        d = j[j["season"] == y]
        out[y] = {b: {"mae_current": round(float(np.abs(d.loc[d["h"].between(*r), "current"] - d.loc[d["h"].between(*r), "actual"]).mean()), 2),
                      "diff": [round(x, 2) for x in boot(d[d["h"].between(*r)], "current", "west")]} for b, r in bands.items()}
        out[y]["rows"] = int(len(d))
    print(json.dumps(out, indent=1))
    (PROC / "west_full.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if "--compare" not in sys.argv:
        run("base", False)
        run("west", True)
    compare()
    if "--compare" not in sys.argv:
        patch(True)
        try:
            print("\nleakage check with the westerly inputs:", flush=True)
            leakage_check.main()
        finally:
            patch(False)
