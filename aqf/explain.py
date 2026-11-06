"""Why the forecast is what it is: SHAP contributions of the hourly model, by group of inputs.

LightGBM's exact TreeSHAP (`pred_contrib`) splits each hour's forecast, in log PM2.5 relative
to the level at the forecast time, into the contributions of the model's ~75 inputs. They are
grouped by what they stand for:

  recent       pollution now and lately: levels, means, trends, the same hour on past days
  season       the seasonal norm: what is typical for the date and hour (climatology), and
               how far the air now is from it
  temperature  air temperature, its daily mean and change, heating degree-days: weather, and
               also the only stand-in for heating demand (emissions are not an input)
  stagnation   what keeps the air from mixing: wind near the ground and aloft, calm hours,
               inversions, pressure, the daily temperature range (clear calm nights)
  wet          rain, snow, humidity, cloud, sunshine
  other        time of day, day of week, CAMS, and the model's average change

For each full forecast day the groups become AQI points: per hour they are applied one after
another, in that order, to the AQI at the forecast time; the AQI steps are averaged over the
day's hours, so they add up exactly to the day's forecast minus the AQI at the forecast time.
This explains the model, not the physics: inputs that move together (temperature and season,
say) share the credit in a way that is partly arbitrary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import aqi
from .features import hourly_rows, pm_state, same_hour_mean

GROUPS = ("recent", "season", "temperature", "stagnation", "wet", "other")
TEMPERATURE = {"w_t2m", "w_t_m24", "w_hdd", "w_dt24", "w_o_t", "dT_o_T"}
WET = {"w_precip", "w_precip24", "w_precip72", "w_snow72", "rain_o_T", "w_rh", "w_rh_m24",
       "w_cloud", "w_cloud_m24", "w_sw", "w_sw24"}


def group_of(col: str) -> str:
    if col in TEMPERATURE:
        return "temperature"
    if col in WET:
        return "wet"
    if col.startswith("w_") or col == "vent_o_T":  # wind, calm, inversions, pressure, daily range
        return "stagnation"
    if col.startswith(("clim", "anom", "doy")):
        return "season"
    if col.startswith("cams_") or col in ("hour", "o_hour", "dow", "h"):
        return "other"
    return "recent"


def contributions(router, origin: pd.Timestamp, pm: pd.Series, wfeat) -> pd.DataFrame:
    """Per horizon h: the start (log PM2.5 at the forecast time) and each group's contribution."""
    lp = np.log1p(pm)
    state, sh7 = pm_state(lp), same_hour_mean(lp)
    parts = []
    for (lo, hi), name in router.choice.items():
        v = router.variants[name]
        H = np.arange(lo, hi + 1)
        X = hourly_rows(pd.DatetimeIndex([origin]), H, lp, state, sh7, v.clim, wfeat, None,
                        cams=getattr(v, "cams", None), lead_matched_vent=True)
        X["src"] = 1.0  # as HourlyModel.predict_frame
        for (blo, bhi), booster in v.boosters.items():
            sel = (H >= blo) & (H <= bhi)
            if not sel.any():
                continue
            cols = v.features if isinstance(v.features, list) else v.features[(blo, bhi)]
            C = booster.predict(X.loc[sel, cols].values, pred_contrib=True)
            g = pd.DataFrame(C[:, :-1], columns=cols).T.groupby([group_of(c) for c in cols]).sum().T
            g["other"] = g.get("other", 0.0) + C[:, -1]  # the model's average change
            g["start"] = X.loc[sel, v.base].values
            g["h"] = H[sel]
            parts.append(g)
    out = pd.concat(parts).set_index("h").sort_index()
    return out.reindex(columns=["start", *GROUPS]).fillna(0.0)


def explain(router, origin: pd.Timestamp, pm: pd.Series, wfeat) -> list[dict]:
    """For each full day within the hourly horizon: AQI now, AQI points per group, the forecast."""
    G = contributions(router, origin, pm, wfeat)
    cum = G["start"].values.copy()
    now = aqi.pm_to_aqi(np.expm1(cum))
    prev, steps = now.copy(), {}
    for g in GROUPS:
        cum = cum + G[g].values
        cur = aqi.pm_to_aqi(np.clip(np.expm1(cum), 0, None))
        steps[g], prev = cur - prev, cur
    t = origin + pd.to_timedelta(G.index, unit="h")
    df = pd.DataFrame({"day": t.normalize(), "forecast": prev, **steps})
    out = []
    for day, d in df.groupby("day"):
        if len(d) < 24:  # today (part already measured) and the last, partial day
            continue
        out.append({"date": f"{day:%Y-%m-%d}", "now": round(float(now[0]), 1),
                    **{g: round(float(d[g].mean()), 1) for g in GROUPS},
                    "forecast": round(float(d["forecast"].mean()), 1)})
    return out
