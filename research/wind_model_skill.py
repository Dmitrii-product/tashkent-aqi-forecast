"""Which weather model forecasts Tashkent's surface wind best? (owner, 2026-11-06: where to get more
accurate wind and humidity forecasts, now or for the future)

research/weather_error_share.py: knowing the wind measured at Tashkent airport would take 13% off the
error 1 day ahead and 19% at 3 days (perfect-weather upper bound), 5-7 points more than knowing the
reanalysis weather. Here: how close each model's archived forecast (Open-Meteo previous runs, lead 1
and 3 days) comes to the airport wind (METAR UZTT, data/raw/weather/metar_UZTT_full.csv), against
our 4-model mean (best match, GFS, ICON, GEM) and the reanalysis. Daily means, October-March from
2024-10 to 2026-11-04; each model over the days its archive covers.

  r_wind   correlation of the daily mean 10 m wind with the airport's
  r_calm   correlation of the daily number of calm hours (< 5 km/h) with the airport's
  rho_aqi  Spearman correlation of the daily mean wind with the daily mean AQI

    PYTHONPATH=. .venv/bin/python research/wind_model_skill.py

Archives are fetched once into data/raw/weather/models_research/<model>_wind_day1_3.parquet.

Result (2026-11-06):
                              lead 1                     lead 3                archive from
                              r_wind  r_calm  rho_aqi    r_wind  r_calm  rho_aqi
  airport (measured)          1.00    1.00    -0.62
  reanalysis (ERA5)           0.72    0.39    -0.55
  our 4-model mean            0.78    0.56    -0.61      0.75    0.53    -0.58   2024-01
  ECMWF IFS 9 km              0.81    0.53    -0.62      0.77    0.51    -0.60   2025-01 (lead 3: 2025-10)
  UK Met Office 10 km         0.73    0.50    -0.55      0.65    0.45    -0.50   2024-08
  GFS                         0.70    0.52    -0.59      0.62    0.43    -0.53   2024-01
  ECMWF IFS 0.25            0.63    0.41    -0.51      0.59    0.38    -0.51   2024-02
  ECMWF AIFS (AI)             0.64    0.52    -0.52      0.63    0.47    -0.55   2025-02
  ICON                        0.64    0.45    -0.52      0.62    0.43    -0.48   2024-01
  CMA GRAPES                  0.60    0.46    -0.48      0.56    0.40    -0.56   2024-01
  GEM                         0.64    0.38    -0.38      0.61    0.32    -0.37   2024-01
  JMA GSM                     0.43    0.21    -0.18      0.38    0.18    -0.13   2024-01
Our mean is about as close to the airport wind as the best single model, ECMWF IFS 9 km — which is
the "best match" member of the mean at this point (same grid cell); on the same days IFS alone is a
little closer (r 0.80 / 0.75 against 0.76 / 0.73 at 1 / 3 days), to be tested once its archive at
3+ days (from 2025-10) covers a winter. Its daily wind tracks the AQI as well as the measured wind. No other global model is closer; the AI models are not better
here. The gap to the measured wind is local (valley and mountain winds, calm under inversions on a
~1-10 km scale) and needs a local high-resolution model. The Tashkent radiosonde (WMO 38457) is in
IGRA only to 2001 and has no current soundings at Wyoming: no measured inversion to use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from aqf.data import RAW, load_prev_runs_ensemble, load_target, load_weather
from aqf.models import daily_aqi

DIR = RAW / "weather" / "models_research"
METAR = RAW / "weather" / "metar_UZTT_full.csv"
MODELS = ["ecmwf_ifs", "ukmo_global_deterministic_10km", "gfs_seamless", "ecmwf_ifs025", "ecmwf_aifs025_single",
          "icon_seamless", "cma_grapes_global", "gem_seamless", "jma_gsm"]
COLD = [10, 11, 12, 1, 2, 3]


def archive(model: str) -> pd.DataFrame:
    path = DIR / f"{model}_wind_day1_3.parquet"
    if not path.exists():  # written once
        r = requests.get("https://previous-runs-api.open-meteo.com/v1/forecast", params={
            "latitude": 41.31, "longitude": 69.28, "models": model, "timezone": "Asia/Tashkent",
            "hourly": "wind_speed_10m_previous_day1,wind_speed_10m_previous_day3",
            "start_date": "2024-01-01", "end_date": "2026-11-04"}, timeout=180)
        r.raise_for_status()
        h = pd.DataFrame(r.json()["hourly"])
        h.index = pd.to_datetime(h.pop("time"))
        h.columns = ["day1" if "day1" in c else "day3" for c in h.columns]
        h.to_parquet(path)
    return pd.read_parquet(path)


def main() -> pd.DataFrame:
    m = pd.read_csv(METAR, na_values="M", usecols=["valid", "sknt"])
    m.index = pd.to_datetime(m.pop("valid"))
    ap = (m["sknt"] * 1.852).resample("h").mean()
    aqi = daily_aqi(load_target()[0]["pm"])
    calm = lambda s: (s < 5).astype(float).where(s.notna()).resample("D").sum(min_count=20)  # noqa: E731

    def stats(s: pd.Series, name: str, lead: int) -> dict:
        s = s.astype(float)
        d = pd.DataFrame({"f": s.resample("D").mean(), "ap": ap.resample("D").mean(), "aqi": aqi,
                          "fc": calm(s), "apc": calm(ap)})
        d = d[d.index.month.isin(COLD) & (d.index >= "2024-10-01")].dropna()
        return {"model": name, "lead": lead, "days": len(d), "from": f"{d.index.min():%Y-%m}",
                "r_wind": round(d["f"].corr(d["ap"]), 2), "r_calm": round(d["fc"].corr(d["apc"]), 2),
                "rho_aqi": round(d["f"].corr(d["aqi"], method="spearman"), 2)}

    rows = [stats(ap, "airport (measured)", 0), stats(load_weather()["wind_speed_10m"], "reanalysis (ERA5)", 0)]
    ens = load_prev_runs_ensemble()
    for lead in (1, 3):
        rows.append(stats(ens[lead]["wind_speed_10m"], "our 4-model mean", lead))
        for model in MODELS:
            col = archive(model)[f"day{lead}"]
            if col.notna().sum() > 24 * 60:
                rows.append(stats(col, model, lead))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    return t


if __name__ == "__main__":
    main()
