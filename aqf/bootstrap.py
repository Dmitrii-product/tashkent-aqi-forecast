"""Download every raw input from scratch (data/ is not in the repository).

    .venv/bin/python -m aqf.bootstrap        # ~30–40 min, most of it the AirNow archive

Then: python -m aqf.backtest && python -m aqf.report && python -m aqf.forecast --retrain
"""
from __future__ import annotations

import pandas as pd

from .data import (EMBASSY, ENSEMBLE_MODELS, IQAIR_JSON, MODELS_DIR, PREV_RUN, RAW, UPPER,
                   WEATHER_ARCHIVE)
from .sources import airnow, cams, openmeteo, tashkentairdata


def main() -> None:
    today = pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None).normalize()
    yesterday = f"{today - pd.Timedelta(days=1):%Y-%m-%d}"
    for d in (RAW, EMBASSY.parent, WEATHER_ARCHIVE.parent, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print("tashkentairdata.uz ...", flush=True)
    tashkentairdata.fetch(IQAIR_JSON)
    print("Open-Meteo archive (ERA5) ...", flush=True)
    openmeteo.cache_archive(WEATHER_ARCHIVE, "2018-01-01", yesterday)
    print("upper-air archive ...", flush=True)
    openmeteo.fetch_upper_history("2021-03-23", yesterday).to_parquet(UPPER)
    print("archived weather forecasts (best match + ensemble models) ...", flush=True)
    for model in [None] + ENSEMBLE_MODELS:
        for d, f in openmeteo.fetch_previous_runs("2024-01-01", f"{today:%Y-%m-%d}", model=model).items():
            path = str(PREV_RUN).format(d) if model is None else MODELS_DIR / f"{model}_day{d}.parquet"
            f.to_parquet(path)
    print("CAMS ...", flush=True)
    cams.update_cache(RAW / "weather" / "cams.parquet")
    print("US Embassy PM2.5 (AirNow file archive, ~25 min) ...", flush=True)
    airnow.download_hourly("2018-06-01", "2025-06-01", EMBASSY)
    print("done")


if __name__ == "__main__":
    main()
