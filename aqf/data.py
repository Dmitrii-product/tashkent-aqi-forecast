"""Assemble the modelling dataset: one hourly PM2.5 series on the IQAir-city scale + weather.

History sources:
  - IQAir city AQI via tashkentairdata.uz        2025-01 .. now   (the real target)
  - US Embassy PM2.5 monitor via AirNow archive   2018 .. 2025-03 (calibrated to IQAir scale
    on the Jan–Mar 2025 overlap, used to extend training/backtest history to 7 winters)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import aqi
from .features import weather_features
from .sources import airnow, tashkentairdata

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

IQAIR_JSON = RAW / "tashkentairdata_air.json"
EMBASSY = RAW / "airnow" / "tashkent_embassy_pm25.parquet"
WEATHER_ARCHIVE = RAW / "weather" / "archive.parquet"


@dataclass
class Calibration:
    a: float  # log(pm_iq) = a + b * log(pm_emb)
    b: float
    n: int
    r: float

    def apply(self, pm_emb: pd.Series) -> pd.Series:
        return np.exp(self.a + self.b * np.log(pm_emb.clip(lower=1.0)))


def load_iqair() -> pd.Series:
    """Hourly PM2.5-equivalent (µg/m³) derived from IQAir US AQI."""
    s = tashkentairdata.load(IQAIR_JSON)
    return pd.Series(aqi.aqi_to_pm(s.values), index=s.index, name="pm")


# Everything below is leak-free by construction: parameters (calibration, diurnal profiles)
# are estimated only on data before `cutoff`, and per-hour transforms look only backwards.
# aqf.leakage_check verifies this by replacing all data after a forecast origin with noise.

def clean_embassy(s: pd.Series) -> pd.Series:
    s = s.copy()
    # single-hour spikes (instrument glitches): > 3x the previous two hours and > 150
    prev = pd.concat([s.shift(1), s.shift(2)], axis=1).max(axis=1)
    s[(s > 3 * prev) & (s > 150)] = np.nan
    # long flat-lines are stuck instruments (counted from the run start, so causal)
    same = s.diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    s[run >= 6] = np.nan
    return s


def calibrate(emb: pd.Series, iq: pd.Series, cutoff: pd.Timestamp | None = None,
              min_days: int = 20) -> Calibration:
    """log-linear map embassy -> IQAir scale from overlapping days before `cutoff`.
    Without enough overlap (backtests before 2025) the embassy series is used as is."""
    df = pd.concat([emb.rename("e"), iq.rename("i")], axis=1).dropna()
    if cutoff is not None:
        df = df[df.index < cutoff]
    # compare 24h means — hour-level noise between a single monitor and a city
    # average is large, but the day-to-day level relationship is stable
    g = df.resample("D")
    d = g.mean()[g.count().min(axis=1) >= 18].dropna()
    if len(d) < min_days:
        return Calibration(a=0.0, b=1.0, n=len(d), r=float("nan"))
    x, y = np.log(d["e"].clip(lower=1)), np.log(d["i"].clip(lower=1))
    b, a = np.polyfit(x, y, 1)
    return Calibration(a=float(a), b=float(b), n=len(d), r=float(np.corrcoef(x, y)[0, 1]))


COLD = (10, 11, 12, 1, 2, 3)


def diurnal_profile(lp: pd.Series) -> dict[bool, np.ndarray]:
    """Mean deviation of lp from its daily mean, by hour, for cold / warm half-years."""
    dev = lp - lp.groupby(lp.index.normalize()).transform("mean")
    cold = lp.index.month.isin(COLD)
    out = {}
    for flag in (True, False):
        d = dev[cold == flag].dropna()
        days = d.index.normalize().nunique()
        out[flag] = d.groupby(d.index.hour).mean().reindex(range(24)).values if days >= 30 else None
    return out


def harmonize_diurnal(emb_lp: pd.Series, iq_lp: pd.Series,
                      cutoff: pd.Timestamp | None = None) -> pd.Series:
    """Give the single embassy monitor the diurnal shape of the IQAir city average.

    The embassy (Yunusabad) peaks in the morning, the city average late in the evening
    (residential heating). Daily levels and weather-driven swings are kept; only the
    within-day shape is swapped, separately for cold and warm half-years. Both profiles
    are estimated on data before `cutoff`.
    """
    past = (lambda s: s) if cutoff is None else (lambda s: s[s.index < cutoff])
    pe, pi = diurnal_profile(past(emb_lp)), diurnal_profile(past(iq_lp))
    out = emb_lp.copy()
    cold = emb_lp.index.month.isin(COLD)
    hr = emb_lp.index.hour.values
    for flag in (True, False):
        if pe[flag] is None or pi[flag] is None:
            continue
        m = cold == flag
        out[m] = emb_lp[m] - pe[flag][hr[m]] + pi[flag][hr[m]]
    return out


def smooth_spikes(lp: pd.Series, a: float) -> pd.Series:
    """Trailing [a, 1-a] smoothing: a single monitor is spikier hour-to-hour than a city
    average of many sensors; left as is it teaches the model to expect every jump to
    revert, which is wrong for the IQAir series we actually forecast."""
    if a <= 0:
        return lp
    sm = (1 - a) * lp + a * lp.shift(1).fillna(lp)
    return sm.where(lp.notna())


EMB_SMOOTH = 0.3


def load_target(profile_cutoff: pd.Timestamp | None = None, smooth: float | None = None,
                iq_aqi: pd.Series | None = None, emb_raw: pd.Series | None = None,
                ) -> tuple[pd.DataFrame, Calibration]:
    """Hourly frame: pm (µg/m³, IQAir scale), src (1 = IQAir, 0 = embassy-calibrated).

    `profile_cutoff` is the forecast origin / training cutoff of a backtest: calibration and
    diurnal profiles are estimated only from data before it. `iq_aqi` / `emb_raw` override
    the raw inputs (used by the leakage check).
    """
    iq_aqi = tashkentairdata.load(IQAIR_JSON) if iq_aqi is None else iq_aqi
    iq = pd.Series(aqi.aqi_to_pm(iq_aqi.values), index=iq_aqi.index, name="pm")
    emb = clean_embassy(airnow.load_hourly(EMBASSY) if emb_raw is None else emb_raw)
    cal = calibrate(emb, iq, profile_cutoff)
    emb_lp = np.log1p(cal.apply(emb.dropna()))
    emb_lp = harmonize_diurnal(emb_lp, np.log1p(iq), profile_cutoff)
    emb_lp = smooth_spikes(emb_lp.asfreq("h"), EMB_SMOOTH if smooth is None else smooth).dropna()
    emb_c = np.expm1(emb_lp)
    idx = pd.date_range(emb_c.index.min(), iq.index.max(), freq="h")
    iq, e = iq.reindex(idx), emb_c.reindex(idx)
    # IQAir wherever it exists; calibrated embassy fills everything else
    df = pd.DataFrame({
        "pm": iq.fillna(e),
        "src": np.where(iq.notna(), 1.0, np.where(e.notna(), 0.0, np.nan)),
    }, index=idx)
    return df, cal


UPPER = RAW / "weather" / "upper.parquet"
PREV_RUN = RAW / "weather" / "prevrun_day{}.parquet"


def load_weather(upper: bool = True) -> pd.DataFrame:
    """Hourly reanalysis weather, plus upper-air temperatures (inversions) where archived."""
    wx = pd.read_parquet(WEATHER_ARCHIVE)
    if upper and UPPER.exists():
        wx = wx.join(pd.read_parquet(UPPER), how="left")
    return wx


def load_prev_runs() -> dict[int, pd.DataFrame]:
    """Archived surface-weather forecasts by lead day (Open-Meteo previous runs, 2024+)."""
    out = {}
    for d in range(1, 8):
        p = Path(str(PREV_RUN).format(d))
        if p.exists():
            out[d] = pd.read_parquet(p)
    return out


MODELS_DIR = RAW / "weather" / "models"
ENSEMBLE_MODELS = ["gfs_seamless", "icon_seamless", "gem_seamless"]


def ensemble_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Average several models' hourly weather; wind direction is averaged as a vector."""
    idx = frames[0].index
    for f in frames[1:]:
        idx = idx.union(f.index)
    frames = [f.reindex(idx) for f in frames]
    num = [c for c in frames[0].columns if c != "wind_direction_10m"]
    out = pd.concat([f[num] for f in frames]).groupby(level=0).mean()
    if "wind_direction_10m" in frames[0]:
        rad = [np.deg2rad(f["wind_direction_10m"]) for f in frames]
        u = pd.concat([np.sin(r) for r in rad], axis=1).mean(axis=1)
        v = pd.concat([np.cos(r) for r in rad], axis=1).mean(axis=1)
        out["wind_direction_10m"] = (np.rad2deg(np.arctan2(u, v)) + 360) % 360
    return out[frames[0].columns]


def load_prev_runs_ensemble(models: list[str] = ENSEMBLE_MODELS) -> dict[int, pd.DataFrame]:
    """Lead-matched archived forecasts averaged over best-match + other global models."""
    base = load_prev_runs()
    out = {}
    for d, f in base.items():
        frames = [f]
        for m in models:
            p = MODELS_DIR / f"{m}_day{d}.parquet"
            if p.exists():
                g = pd.read_parquet(p)
                if len(g):
                    frames.append(g)
        out[d] = ensemble_mean(frames)
    return out


def splice_forecast_weather(archive: pd.DataFrame, prev: dict[int, pd.DataFrame]) -> dict[int, pd.DataFrame]:
    """Per lead day: archived forecast where available, reanalysis elsewhere (pre-2024).

    The model is trained on this too, so it learns how forecast weather at each lead maps
    to what then happens (Open-Meteo's forecast wind is weaker than ERA5, and the gap drifts
    with model upgrades; quantile-mapping it on past windows over-corrected)."""
    out = {}
    for d, f in prev.items():
        w = archive.astype(float)
        common = w.index.intersection(f.index)
        cols = list(f.columns)
        # forecast where it has a value, reanalysis otherwise
        w.loc[common, cols] = f.loc[common, cols].astype(float).fillna(w.loc[common, cols]).values
        out[d] = weather_features(w)
    return out


def iqair_meta() -> dict:
    return json.loads(IQAIR_JSON.read_text())
