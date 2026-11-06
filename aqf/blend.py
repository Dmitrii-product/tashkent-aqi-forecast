"""The served hourly forecast since model version 2026-11-04: LightGBM for the first hours, then
the 50/50 blend (in log PM2.5) with the neural network of aqf.nn.

No PyTorch here: aqf.forecast imports this module and runs the network in its own process
(`python -m aqf.nn serve`), so the forecast still goes out, from LightGBM alone, if it fails.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import aqi

BLEND_FROM_H = 7  # the blend starts here: at 1-3 h the 50/50 blend lost to LightGBM in the backtest


def blend_lp(gbm_lp: np.ndarray, nn_lp: np.ndarray, h: np.ndarray) -> np.ndarray:
    return np.where(h >= BLEND_FROM_H, 0.5 * (gbm_lp + nn_lp), gbm_lp)


def to_aqi(lp: np.ndarray) -> np.ndarray:
    return aqi.pm_to_aqi(np.clip(np.expm1(lp), 0, None))


def nn_record(origin: str, issued_at: str, generated_at: str, nn_lp: np.ndarray, gbm_lp: np.ndarray,
              backfilled: bool, seeds: list[int]) -> dict:
    """The <stem>_nn.json of an issued forecast: the network and the blend, hour by hour and per
    full day, next to LightGBM (scored on the other-forecasts tab, aqf.live.shadow)."""
    o = pd.Timestamp(origin)
    h = np.arange(1, len(nn_lp) + 1)
    t = o + pd.to_timedelta(h, unit="h")
    nn_aqi, blend_aqi = to_aqi(nn_lp), to_aqi(blend_lp(gbm_lp, nn_lp, h))
    hourly = [{"t": f"{ti:%Y-%m-%dT%H:%M:%S}", "nn": round(float(a)), "blend": round(float(b))}
              for ti, a, b in zip(t, nn_aqi, blend_aqi)]
    df = pd.DataFrame({"day": t.normalize(), "nn": nn_aqi, "blend": blend_aqi})
    g = df.groupby("day").agg(n=("nn", "size"), nn=("nn", "mean"), blend=("blend", "mean"))
    daily = [{"t": f"{d:%Y-%m-%d}", "nn": round(float(r.nn)), "blend": round(float(r.blend))}
             for d, r in g[g["n"] == 24].iterrows()]
    return {"origin": origin, "issued_at": issued_at, "generated_at": generated_at, "backfilled": backfilled,
            "seeds": list(seeds), "blend_from_h": BLEND_FROM_H, "hourly_7d": hourly, "daily": daily}
