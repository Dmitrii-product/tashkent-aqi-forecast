"""Feature engineering for the hourly (direct multi-horizon) and daily models.

All timestamps are naive Tashkent local time (UTC+5, no DST).
`lp` everywhere means log1p(PM2.5 µg/m³).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

CALM_KMH = 5.0


# --------------------------------------------------------------------------- weather
def weather_features(w: pd.DataFrame) -> pd.DataFrame:
    """Per-hour weather descriptors; rolling windows look back from each hour."""
    w = w.asfreq("h").interpolate(limit=3)
    ws = w["wind_speed_10m"]
    t = w["temperature_2m"]
    wd = np.deg2rad(w["wind_direction_10m"])
    f = pd.DataFrame(index=w.index)
    f["t2m"] = t
    f["rh"] = w["relative_humidity_2m"]
    f["ws"] = ws
    f["gust"] = w["wind_gusts_10m"]
    # wind blowing *from* direction -> u/v components scaled by speed
    f["u"] = -ws * np.sin(wd)
    f["v"] = -ws * np.cos(wd)
    f["precip"] = w["precipitation"]
    f["cloud"] = w["cloud_cover"]
    f["psfc"] = w["surface_pressure"]
    f["sw"] = w["shortwave_radiation"]
    f["ws_m6"] = ws.rolling(6, min_periods=3).mean()
    f["ws_m24"] = ws.rolling(24, min_periods=12).mean()
    f["gust_max24"] = w["wind_gusts_10m"].rolling(24, min_periods=12).max()
    f["calm24"] = (ws < CALM_KMH).astype(float).rolling(24, min_periods=12).sum()
    f["precip24"] = w["precipitation"].rolling(24, min_periods=12).sum()
    f["precip72"] = w["precipitation"].rolling(72, min_periods=36).sum()
    f["snow72"] = w["snowfall"].rolling(72, min_periods=36).sum()
    f["t_m24"] = t.rolling(24, min_periods=12).mean()
    f["t_rng24"] = t.rolling(24, min_periods=12).max() - t.rolling(24, min_periods=12).min()
    f["hdd"] = (18.0 - f["t_m24"]).clip(lower=0)
    f["dt24"] = t - t.shift(24)
    f["dp24"] = w["surface_pressure"] - w["surface_pressure"].shift(24)
    f["rh_m24"] = w["relative_humidity_2m"].rolling(24, min_periods=12).mean()
    f["cloud_m24"] = w["cloud_cover"].rolling(24, min_periods=12).mean()
    f["sw24"] = w["shortwave_radiation"].rolling(24, min_periods=12).sum()
    if "temperature_850hPa" in w:
        # inversion strength: positive = warmer air aloft than at the surface
        f["dT925"] = w["temperature_925hPa"] - t
        f["dT850"] = w["temperature_850hPa"] - t
        f["dT700"] = w["temperature_700hPa"] - t
        f["dT850_max24"] = f["dT850"].rolling(24, min_periods=12).max()
        f["dT850_m24"] = f["dT850"].rolling(24, min_periods=12).mean()
        f["ws850"] = w["wind_speed_850hPa"]
        wd8 = np.deg2rad(w["wind_direction_850hPa"])
        f["u850"] = -w["wind_speed_850hPa"] * np.sin(wd8)
        f["v850"] = -w["wind_speed_850hPa"] * np.cos(wd8)
    # how the wind blew over the last day: the mean westerly component and the strongest gust
    # from the west (225–315°). Westerly flow here means fronts and clean air (since 2026-10-28,
    # research/westerly_wind_full.py); appended last, as in that test
    f["u_m24"] = f["u"].rolling(24, min_periods=12).mean()
    west = w["wind_direction_10m"].between(225, 315)
    f["gust_w_max24"] = w["wind_gusts_10m"].where(west, 0.0).rolling(24, min_periods=12).max()
    return f




def daily_weather(w: pd.DataFrame) -> pd.DataFrame:
    w = w.asfreq("h").interpolate(limit=3)
    g = w.resample("D")
    d = pd.DataFrame({
        "t_mean": g["temperature_2m"].mean(),
        "t_min": g["temperature_2m"].min(),
        "t_max": g["temperature_2m"].max(),
        "ws_mean": g["wind_speed_10m"].mean(),
        "ws_max": g["wind_speed_10m"].max(),
        "gust_max": g["wind_gusts_10m"].max(),
        "calm_h": g["wind_speed_10m"].apply(lambda x: float((x < CALM_KMH).sum())),
        "precip": g["precipitation"].sum(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "cloud_mean": g["cloud_cover"].mean(),
        "psfc_mean": g["surface_pressure"].mean(),
    })
    d["t_rng"] = d["t_max"] - d["t_min"]
    d["hdd"] = (18.0 - d["t_mean"]).clip(lower=0)
    d["precip_prev"] = d["precip"].shift(1)
    d["dp"] = d["psfc_mean"].diff()
    return d


# --------------------------------------------------------------------------- climatology
def season_of(ts: pd.DatetimeIndex) -> np.ndarray:
    """Season label = year in which the Jul–Jun season starts (Oct 2025 -> 2025)."""
    return np.where(ts.month >= 7, ts.year, ts.year - 1)


def _circ_smooth(a: np.ndarray, win: int, axis: int) -> np.ndarray:
    """Circular moving sum of width 2*win+1 along `axis` (day-of-year)."""
    a = np.moveaxis(a, axis, -1)
    ext = np.concatenate([a[..., -win:], a, a[..., :win]], axis=-1)
    cs = np.cumsum(np.concatenate([np.zeros(a.shape[:-1] + (1,)), ext], axis=-1), axis=-1)
    out = cs[..., 2 * win + 1:] - cs[..., : -(2 * win + 1)]
    return np.moveaxis(out, -1, axis)


class Climatology:
    """Mean lp by (day-of-year, hour), circularly smoothed over ±`win` days.

    Keeps per-season sums so that training rows can look up a leave-own-season-out
    climatology: otherwise with only a handful of years the "climatology" of a training
    target is largely a smoothed copy of that same target (leakage that does not exist
    at forecast time, when the future season is never part of the climatology).
    """

    MIN_COUNT = 20

    def __init__(self, win: int = 15):
        self.win = win

    def fit(self, lp: pd.Series) -> "Climatology":
        lp = lp.dropna()
        seas = season_of(lp.index)
        self.seasons = np.unique(seas)
        yi = np.searchsorted(self.seasons, seas)
        doy = np.minimum(lp.index.dayofyear.values, 365) - 1
        hr = lp.index.hour.values
        S = np.zeros((len(self.seasons), 365, 24))
        C = np.zeros_like(S)
        np.add.at(S, (yi, doy, hr), lp.values)
        np.add.at(C, (yi, doy, hr), 1)
        self.S = _circ_smooth(S, self.win, axis=1)
        self.C = _circ_smooth(C, self.win, axis=1)
        self.St, self.Ct = self.S.sum(0), self.C.sum(0)
        tab = self.St / np.maximum(self.Ct, 1)
        tab[self.Ct < self.MIN_COUNT] = np.nan
        tab = pd.DataFrame(tab).interpolate(limit_direction="both").values
        self.table = tab
        self.daily = tab.mean(axis=1)
        return self

    def _idx(self, ts: pd.DatetimeIndex):
        return np.minimum(ts.dayofyear.values, 365) - 1, ts.hour.values

    def at(self, ts: pd.DatetimeIndex, excl: np.ndarray | None = None) -> np.ndarray:
        d, h = self._idx(ts)
        if excl is None:
            return self.table[d, h]
        pos = np.searchsorted(self.seasons, excl)
        pos_c = np.clip(pos, 0, len(self.seasons) - 1)
        own = (pos < len(self.seasons)) & (self.seasons[pos_c] == excl)
        s = self.St[d, h] - np.where(own, self.S[pos_c, d, h], 0.0)
        c = self.Ct[d, h] - np.where(own, self.C[pos_c, d, h], 0.0)
        return np.where(c >= self.MIN_COUNT, s / np.maximum(c, 1), np.nan)

    def day(self, ts: pd.DatetimeIndex, excl: np.ndarray | None = None) -> np.ndarray:
        if excl is None:
            return self.daily[np.minimum(ts.dayofyear.values, 365) - 1]
        hours = [ts.normalize() + pd.Timedelta(hours=k) for k in range(24)]
        with warnings.catch_warnings():  # all-NaN days early in the record
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(np.stack([self.at(t, excl) for t in hours]), axis=0)


# --------------------------------------------------------------------------- pm state at origin
def pm_state(lp: pd.Series) -> pd.DataFrame:
    """Features describing recent pollution, indexed by the origin hour."""
    s = pd.DataFrame(index=lp.index)
    cur = lp.ffill(limit=2)  # tolerate 1–2 h reporting gaps at the origin
    s["lag0"] = cur
    for k in (1, 2, 3, 6, 12, 24, 48):
        s[f"lag{k}"] = lp.shift(k)
    for k, mp in ((3, 2), (6, 4), (24, 16), (72, 48), (168, 100)):
        s[f"m{k}"] = lp.rolling(k, min_periods=mp).mean()
    s["max24"] = lp.rolling(24, min_periods=16).max()
    s["min24"] = lp.rolling(24, min_periods=16).min()
    s["sd24"] = lp.rolling(24, min_periods=16).std()
    s["d1"] = s["lag0"] - s["lag1"]
    s["d3"] = s["lag0"] - s["lag3"]
    s["d24"] = s["m24"] - s["m24"].shift(24)
    return s


def same_hour_mean(lp: pd.Series, days: int = 7) -> pd.Series:
    """Mean of lp at the same clock hour over the last `days` days (incl. this hour)."""
    stack = pd.concat([lp.shift(24 * k) for k in range(days)], axis=1)
    return stack.mean(axis=1, skipna=True).where(stack.notna().sum(axis=1) >= days // 2)


# --------------------------------------------------------------------------- hourly rows
def lead_matched_vent_rain(o: pd.DatetimeIndex, h: np.ndarray,
                           wfeat: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    """Mean wind and rain sum over hours o+1..o+h, each hour o+k from the frame of lead day
    (k-1)//24 + 1 (the forecast issued by the origin), as `vent_o_T` / `rain_o_T`."""
    ws, pr = np.zeros(len(o)), np.zeros(len(o))
    top = max(wfeat)
    for L, f in wfeat.items():
        s = 24 * (L - 1)
        e = np.clip(h, s, 24 * L if L < top else max(int(h.max()), 24 * L))
        m = e > s
        if not m.any():
            continue
        cw = f["ws"].fillna(f["ws"].mean()).cumsum()
        cp = f["precip"].fillna(0).cumsum()
        te, ts = o[m] + pd.to_timedelta(e[m], unit="h"), o[m] + pd.Timedelta(hours=s)
        ws[m] += cw.reindex(te).values - cw.reindex(ts).values
        pr[m] += cp.reindex(te).values - cp.reindex(ts).values
    return ws / h, pr


def hourly_rows(origins: pd.DatetimeIndex, horizons: np.ndarray, lp: pd.Series,
                state: pd.DataFrame, sh7: pd.Series, clim: Climatology,
                wfeat: pd.DataFrame | dict[int, pd.DataFrame],
                src: pd.Series | None = None, loo: bool = False,
                cams: pd.DataFrame | None = None, cams_max_h: int = 24,
                lead_matched_vent: bool = False) -> pd.DataFrame:
    """Design matrix for (origin, horizon) pairs.

    `horizons` is either 1-D (same horizons for every origin) or 2-D [n_origins, k].
    `wfeat` is an hourly weather-feature frame, or {lead_day: frame} to use archived
    forecasts with the lead time matching each row.

    `lead_matched_vent` (only matters for a {lead_day: frame} dict): sum the ventilation and
    wash-out between origin and target hour o+k from the frame of its own lead day
    ((k-1)//24 + 1), i.e. only from forecasts issued by the origin. Scoring with archived
    forecasts (HourlyModel.predict_frame: backtest, leakage check) uses it; training keeps the
    lead-1 frame as the served models were trained (research/forecast_trust_test.py, 2026-11-05:
    training on lead-matched rows was not better). With one issued-forecast frame, as in
    production, both ways are the same.
    """
    n = len(origins)
    H = np.broadcast_to(horizons, (n, horizons.shape[-1])) if horizons.ndim == 1 else horizons
    k = H.shape[1]
    o = pd.DatetimeIndex(np.repeat(origins.values, k))
    h = H.reshape(-1).astype(int)
    T = o + pd.to_timedelta(h, unit="h")

    X = pd.DataFrame(state.reindex(o).values, columns=state.columns)
    X["h"] = h
    X["hour"] = T.hour
    X["dow"] = T.dayofweek
    X["o_hour"] = o.hour
    doy = T.dayofyear.values
    X["doy_s"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_c"] = np.cos(2 * np.pi * doy / 365.25)
    excl = season_of(o) if loo else None
    X["clim_T"] = clim.at(T, excl)
    X["clim_o"] = clim.at(o, excl)
    X["clim_day_o"] = clim.day(o, excl)
    X["anom0"] = X["lag0"] - X["clim_o"]
    X["anom24"] = X["m24"] - X["clim_day_o"]
    X["anom168"] = X["m168"] - X["clim_day_o"]
    X["clim_T_minus_lag0"] = X["clim_T"] - X["lag0"]
    X["clim_T_minus_o"] = X["clim_T"] - X["clim_o"]
    back = pd.to_timedelta(24 * np.ceil(h / 24), unit="h")
    X["same_hour_last"] = lp.reindex(T - back).values
    X["same_hour_7d"] = sh7.reindex(T - back).values

    if isinstance(wfeat, dict):
        lead = np.clip((h - 1) // 24 + 1, 1, max(wfeat))
        cols = next(iter(wfeat.values())).columns
        W = np.full((len(T), len(cols)), np.nan)
        for L, frame in wfeat.items():
            m = lead == L
            if m.any():
                W[m] = frame.reindex(T[m]).values
        wf_T = pd.DataFrame(W, columns=cols)
        base = wfeat[1]
    else:
        wf_T = pd.DataFrame(wfeat.reindex(T).values, columns=wfeat.columns)
        base = wfeat
    X = pd.concat([X, wf_T.add_prefix("w_")], axis=1)
    # ventilation and wash-out between origin and target
    cs_ws = base["ws"].fillna(base["ws"].mean()).cumsum()
    cs_pr = base["precip"].fillna(0).cumsum()
    X["vent_o_T"] = (cs_ws.reindex(T).values - cs_ws.reindex(o).values) / h
    X["rain_o_T"] = cs_pr.reindex(T).values - cs_pr.reindex(o).values
    if lead_matched_vent and isinstance(wfeat, dict):
        X["vent_o_T"], X["rain_o_T"] = lead_matched_vent_rain(o, h, wfeat)
    X["w_o_ws24"] = base["ws_m24"].reindex(o).values
    X["w_o_t"] = base["t2m"].reindex(o).values
    X["dT_o_T"] = X["w_t2m"] - X["w_o_t"]

    if cams is not None:
        # CAMS PM2.5 (log), its 24 h mean and change since the origin; only within the
        # horizon where the stitched CAMS series is a fair stand-in for a live forecast
        c = np.log1p(cams["pm2_5"].asfreq("h"))
        c24 = c.rolling(24, min_periods=12).mean()
        keep = h <= cams_max_h
        X["cams_T"] = np.where(keep, c.reindex(T).values, np.nan)
        X["cams_m24_T"] = np.where(keep, c24.reindex(T).values, np.nan)
        X["cams_d"] = X["cams_T"] - c.reindex(o).values
        X["cams_dust_T"] = np.where(keep, np.log1p(cams["dust"].asfreq("h")).reindex(T).values, np.nan)
    X["src"] = 1.0 if src is None else src.reindex(T).values
    X.index = pd.MultiIndex.from_arrays([o, h], names=["origin", "h"])
    return X


def target_times(X: pd.DataFrame) -> pd.DatetimeIndex:
    o = X.index.get_level_values("origin")
    h = X.index.get_level_values("h")
    return o + pd.to_timedelta(h, unit="h")
