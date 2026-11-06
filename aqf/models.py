"""Forecast models.

HourlyModel — direct multi-horizon gradient boosting: one booster per horizon group,
              predicting the change of log1p(PM2.5) from the last observed hour.
DailyModel  — same idea on daily mean AQI for horizons 1..90 days. Weather is used for
              the first `wx_days` days (forecast); afterwards climatological normals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import aqi
from .features import (Climatology, _circ_smooth, daily_weather, hourly_rows, pm_state,
                       same_hour_mean, season_of, target_times)

COLD_MONTHS = {10, 11, 12, 1, 2, 3}

# Versions of the forecast model, oldest first: a new one whenever a change alters what the
# model forecasts (inputs, algorithms), not for the weekly retrain on fresh data. aqf.forecast
# retrains when models.pkl comes from another version and stamps every issued forecast with
# it; the site lists the versions with the first forecast each one issued (texts in the
# dashboard template under the same id). `backtest`: MAE change over 1–168 h against the
# previous version under the full protocol, per period (choice / unseen / unseen).
MODEL_VERSIONS = [
    {"id": "2026-10-18"},
    {"id": "2026-10-28", "inputs": ["u_m24", "gust_w_max24"],
     "backtest": [{"period": "2024/25", "mae": 32.51, "diff": -0.39, "ci": [-0.57, -0.22]},
                  {"period": "2025/26", "mae": 31.53, "diff": -0.57, "ci": [-0.74, -0.40]},
                  {"period": "2026-10", "mae": 26.18, "diff": -0.23, "ci": [-0.69, 0.20]}]},
    # the LightGBM models of 2026-10-28 unchanged; from hour 7 on the forecast is their 50/50 blend
    # (log PM2.5) with the network of aqf.nn (aqf.blend); backtest on the pairs of aqf.nn's test
    {"id": "2026-11-04", "gbm": "2026-10-28", "blend": "nn",
     "backtest": [{"period": "2024/25", "mae": 32.11, "diff": -0.46, "ci": [-0.78, -0.17]},
                  {"period": "2025/26", "mae": 30.95, "diff": -1.03, "ci": [-1.44, -0.63]},
                  {"period": "2026-10", "mae": 26.33, "diff": -2.89, "ci": [-3.90, -1.78]}]},
]
MODEL_VERSION = MODEL_VERSIONS[-1]["id"]
FIRST_VERSION = MODEL_VERSIONS[0]["id"]  # forecasts issued before the stamp was added
# the version of the LightGBM models in models.pkl: a version that only adds the network keeps them
GBM_VERSION = MODEL_VERSIONS[-1].get("gbm", MODEL_VERSION)
# what an issued forecast is stamped with when the network failed and LightGBM went out alone
GBM_ONLY_VERSION = next(v["id"] for v in reversed(MODEL_VERSIONS) if "blend" not in v)

HOURLY_PARAMS = dict(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                     l2_regularization=10.0, max_features=0.7)
DAILY_PARAMS = dict(learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=300,
                    l2_regularization=10.0, max_features=0.7)


def _gbm(params: dict, rounds: int, seed: int, X: pd.DataFrame, y, w, algo: str = "hgb",
         extra: dict | None = None):
    """Fit one regressor. hgb = scikit-learn HistGradientBoosting; xgb / lgbm need an OpenMP
    runtime (run with DYLD_FALLBACK_LIBRARY_PATH=<sklearn>/.dylibs on this Mac)."""
    if algo == "hgb":
        m = HistGradientBoostingRegressor(max_iter=rounds, early_stopping=False,
                                          random_state=seed, **params)
    elif algo == "xgb":
        import xgboost
        m = xgboost.XGBRegressor(
            n_estimators=rounds, learning_rate=params["learning_rate"], tree_method="hist",
            grow_policy="lossguide", max_depth=0, max_leaves=params["max_leaf_nodes"],
            min_child_weight=params["min_samples_leaf"] / 50, subsample=0.8,
            colsample_bytree=params["max_features"], reg_lambda=params["l2_regularization"],
            n_jobs=8, random_state=seed)
    elif algo == "lgbm":
        try:
            import lightgbm
        except OSError:  # no OpenMP runtime on this machine -> same algorithm family
            import warnings
            warnings.warn("LightGBM unavailable (libomp); falling back to HistGradientBoosting")
            return _gbm(params, rounds, seed, X, y, w, "hgb")
        kw = dict(
            n_estimators=rounds, learning_rate=params["learning_rate"],
            num_leaves=params["max_leaf_nodes"], min_child_samples=params["min_samples_leaf"],
            subsample=0.8, subsample_freq=1, colsample_bytree=params["max_features"],
            reg_lambda=params["l2_regularization"], n_jobs=8, random_state=seed, verbose=-1)
        kw.update(extra or {})  # e.g. objective="l1", num_leaves=63
        m = lightgbm.LGBMRegressor(**kw)
    elif algo in ("ridge", "et", "mlp"):
        # other model families; they need imputed (and, for ridge/mlp, scaled) inputs
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        steps = [("imp", SimpleImputer(strategy="median", keep_empty_features=True))]
        if algo == "ridge":
            steps += [("sc", StandardScaler()), ("model", Ridge(alpha=10.0))]
        elif algo == "et":
            steps += [("model", ExtraTreesRegressor(n_estimators=120, min_samples_leaf=40,
                                                    max_features=0.5, n_jobs=8, random_state=seed))]
        else:
            steps += [("sc", StandardScaler()),
                      ("model", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                             learning_rate_init=1e-3, max_iter=60, batch_size=512,
                                             early_stopping=True, n_iter_no_change=6,
                                             random_state=seed))]
        m = Pipeline(steps)
        try:
            return m.fit(X.values, y, model__sample_weight=w)
        except TypeError:  # estimator without sample_weight support
            return m.fit(X.values, y)
    else:
        raise ValueError(algo)
    return m.fit(X.values, y, sample_weight=w)


def hampel(lp: pd.Series, window: int = 7, k: float = 3.0) -> pd.Series:
    """Replace isolated spikes (> k robust SDs from the centred rolling median) by the median."""
    med = lp.rolling(window, center=True, min_periods=3).median()
    mad = (lp - med).abs().rolling(window, center=True, min_periods=3).median() * 1.4826
    bad = (lp - med).abs() > k * mad.clip(lower=0.05)
    return lp.where(~bad, med)


IQAIR_WEIGHT = 3.0


def _weights(T: pd.DatetimeIndex, src: np.ndarray) -> np.ndarray:
    cold = np.isin(T.month, list(COLD_MONTHS))
    return (1.0 + cold) * (1.0 + (IQAIR_WEIGHT - 1.0) * np.nan_to_num(src, nan=0.0))


# Horizon groups, each with its own booster. A single model with horizon as a feature
# lets the long horizons (much larger target variance) dominate the loss and smears the
# first hours — in testing it lost to plain persistence at h=1..3.
H_GROUPS = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 18), (19, 24),
            (25, 36), (37, 48), (49, 72), (73, 96), (97, 120), (121, 144), (145, 168)]


class HourlyModel:
    def __init__(self, max_h: int = 168, origin_step: int = 2, per_origin: int = 4,
                 rounds: int = 300, seed: int = 0, base: str = "lag0", algo: str = "lgbm",
                 half_life_days: float | None = None, train_start: pd.Timestamp | None = None,
                 cams: pd.DataFrame | None = None, lgbm_extra: dict | None = None,
                 clean_target: bool = False, only_groups: list[tuple[int, int]] | None = None):
        self.base = base  # what the GBM predicts the deviation from: lag0 | m24 | clim_T
        self.algo = algo
        self.half_life_days = half_life_days  # down-weight old years (recency weighting)
        self.train_start = train_start  # ignore training targets before this date
        self.cams = cams  # optional CAMS frame -> extra features for horizons <= 24 h
        self.lgbm_extra = lgbm_extra or {}
        self.clean_target = clean_target  # Hampel-filter isolated spikes out of training targets
        self.max_h = max_h
        self.groups = [g for g in H_GROUPS if g[0] <= max_h and (only_groups is None or g in only_groups)]
        self.origin_step = origin_step
        self.per_origin = per_origin
        self.rounds = rounds
        self.seed = seed

    # ---------------------------------------------------------------- training
    def fit(self, pm: pd.Series, src: pd.Series, wfeat: pd.DataFrame,
            cutoff: pd.Timestamp | None = None) -> "HourlyModel":
        lp = np.log1p(pm)
        if cutoff is not None:
            lp = lp[lp.index < cutoff]
        self.clim = Climatology().fit(lp)
        state = pm_state(lp)
        sh7 = same_hour_mean(lp)
        rng = np.random.default_rng(self.seed)
        origins = state.index[(state.index.hour % self.origin_step == 0) & state["lag0"].notna()]
        if self.train_start is not None:
            origins = origins[origins >= self.train_start]
        end = lp.index.max()
        ly = hampel(lp) if self.clean_target else lp
        self.boosters = {}
        self.features = {}
        self.n_train = 0
        for lo, hi in self.groups:
            k = min(self.per_origin, hi - lo + 1)
            H = rng.integers(lo, hi + 1, size=(len(origins), k)) if hi > lo else np.full((len(origins), 1), lo)
            X = hourly_rows(origins, H, lp, state, sh7, self.clim, wfeat, src, loo=True,
                            cams=self.cams)
            T = target_times(X)
            y = ly.reindex(T).values - X[self.base].values
            ok = ~np.isnan(y)
            X, y, T = X[ok], y[ok], T[ok]
            X = X.loc[:, X.notna().any()]  # e.g. CAMS columns beyond 24 h
            self.features[(lo, hi)] = list(X.columns)
            w = _weights(T, X["src"].values)
            if self.half_life_days:
                w = w * 0.5 ** ((end - T) / pd.Timedelta(days=self.half_life_days))
            self.boosters[(lo, hi)] = _gbm(HOURLY_PARAMS, self.rounds, self.seed, X, y,
                                           np.asarray(w, float), self.algo, self.lgbm_extra)
            self.n_train += len(y)
        return self

    # ---------------------------------------------------------------- inference
    def predict_frame(self, origins: pd.DatetimeIndex, pm: pd.Series,
                      wfeat: pd.DataFrame | dict, horizons: np.ndarray | None = None) -> pd.DataFrame:
        """Frame indexed by (origin, h). `pm` may extend past the origins: every feature
        only reads values at or before its origin."""
        lp = np.log1p(pm)
        state = pm_state(lp)
        sh7 = same_hour_mean(lp)
        H = np.arange(1, self.max_h + 1) if horizons is None else horizons
        X = hourly_rows(origins, H, lp, state, sh7, self.clim, wfeat, None,
                        cams=getattr(self, "cams", None), lead_matched_vent=True)
        X["src"] = 1.0
        h = X.index.get_level_values("h").values
        z = np.full(len(X), np.nan)
        for (lo, hi), b in self.boosters.items():
            m = (h >= lo) & (h <= hi)
            if m.any():
                cols = self.features if isinstance(self.features, list) else self.features[(lo, hi)]
                z[m] = b.predict(X.loc[m, cols].values)
        out = pd.DataFrame(index=X.index)
        out["pred_lp"] = z + X[self.base].values
        out["pred_pm"] = np.expm1(out["pred_lp"]).clip(lower=0)
        out["pred_aqi"] = aqi.pm_to_aqi(out["pred_pm"].values)
        out["clim_aqi"] = aqi.pm_to_aqi(np.expm1(X["clim_T"].values))
        out["persist_aqi"] = aqi.pm_to_aqi(np.expm1(X["lag0"].values))
        out["same_hour_aqi"] = aqi.pm_to_aqi(np.expm1(X["same_hour_last"].values))
        return out


# =========================================================================== daily
def daily_aqi(pm: pd.Series, min_hours: int = 18) -> pd.Series:
    """Daily mean of hourly AQI (what tashkentairdata.uz shows on its calendar)."""
    a = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    g = a.resample("D")
    return g.mean().where(g.count() >= min_hours)


class SeasonalMean:
    """Day-of-year mean of a daily series, smoothed ±win days, with leave-one-season-out."""

    MIN_COUNT = 8

    def __init__(self, win: int = 10):
        self.win = win

    def fit(self, s: pd.Series) -> "SeasonalMean":
        s = s.dropna()
        seas = season_of(s.index)
        self.seasons = np.unique(seas)
        yi = np.searchsorted(self.seasons, seas)
        doy = np.minimum(s.index.dayofyear.values, 365) - 1
        S = np.zeros((len(self.seasons), 365))
        C = np.zeros_like(S)
        np.add.at(S, (yi, doy), s.values)
        np.add.at(C, (yi, doy), 1)
        self.S, self.C = _circ_smooth(S, self.win, 1), _circ_smooth(C, self.win, 1)
        self.St, self.Ct = self.S.sum(0), self.C.sum(0)
        tab = self.St / np.maximum(self.Ct, 1)
        tab[self.Ct < self.MIN_COUNT] = np.nan
        self.table = pd.Series(tab).interpolate(limit_direction="both").values
        return self

    def at(self, days: pd.DatetimeIndex, excl: np.ndarray | None = None) -> np.ndarray:
        d = np.minimum(days.dayofyear.values, 365) - 1
        if excl is None:
            return self.table[d]
        pos = np.searchsorted(self.seasons, excl)
        pc = np.clip(pos, 0, len(self.seasons) - 1)
        own = (pos < len(self.seasons)) & (self.seasons[pc] == excl)
        num = self.St[d] - np.where(own, self.S[pc, d], 0.0)
        den = self.Ct[d] - np.where(own, self.C[pc, d], 0.0)
        return np.where(den >= self.MIN_COUNT, num / np.maximum(den, 1), self.table[d])


class DailyClim:
    """Climatology of log daily AQI plus weather normals (all leave-one-season-out capable)."""

    def fit(self, la: pd.Series, dw: pd.DataFrame) -> "DailyClim":
        self.la = SeasonalMean().fit(la)
        self.wx = {c: SeasonalMean().fit(dw[c]) for c in dw.columns}
        return self

    def at(self, days: pd.DatetimeIndex, excl=None) -> np.ndarray:
        return self.la.at(days, excl)

    def wx_at(self, days: pd.DatetimeIndex, excl=None) -> pd.DataFrame:
        return pd.DataFrame({c: m.at(days, excl) for c, m in self.wx.items()})


class DailyModel:
    def __init__(self, max_d: int = 90, wx_days: int = 7, rounds: int = 500, seed: int = 0):
        self.max_d = max_d
        self.wx_days = wx_days
        self.rounds = rounds
        self.seed = seed

    def _state(self, la: pd.Series, loo: bool = False) -> pd.DataFrame:
        an = la - self.clim.at(la.index, season_of(la.index) if loo else None)
        s = pd.DataFrame(index=la.index)
        s["an1"] = an
        for k in (3, 7, 14, 30, 90):
            s[f"an{k}"] = an.rolling(k, min_periods=max(1, int(k * 0.6))).mean()
        s["an365"] = an.rolling(365, min_periods=150).mean()
        s["lvl7"] = la.rolling(7, min_periods=4).mean()
        s["an1"] = s["an1"].fillna(s["an3"])
        return s

    def _rows(self, origins: pd.DatetimeIndex, state: pd.DataFrame, dw: pd.DataFrame | None,
              loo: bool = False) -> pd.DataFrame:
        D = np.arange(1, self.max_d + 1)
        o = pd.DatetimeIndex(np.repeat(origins.values, len(D)))
        d = np.tile(D, len(origins))
        T = o + pd.to_timedelta(d, unit="D")
        X = pd.DataFrame(state.reindex(o).values, columns=state.columns)
        X["d"] = d
        X["dow"] = T.dayofweek
        doy = T.dayofyear.values
        X["doy_s"] = np.sin(2 * np.pi * doy / 365.25)
        X["doy_c"] = np.cos(2 * np.pi * doy / 365.25)
        excl = season_of(o) if loo else None
        X["clim_T"] = self.clim.at(T, excl)
        normals = self.clim.wx_at(T, excl)
        if dw is not None:
            actual = pd.DataFrame(dw.reindex(T).values, columns=dw.columns)
            use = (d <= self.wx_days)[:, None] & actual.notna().values
            wx = pd.DataFrame(np.where(use, actual.values, normals[dw.columns].values),
                              columns=dw.columns)
        else:
            wx = normals
        X = pd.concat([X, wx.add_prefix("w_")], axis=1)
        X["wx_known"] = (d <= self.wx_days).astype(float)
        X.index = pd.MultiIndex.from_arrays([o, d], names=["origin", "d"])
        return X

    def fit(self, pm: pd.Series, weather: pd.DataFrame, cutoff: pd.Timestamp | None = None):
        a = daily_aqi(pm)
        if cutoff is not None:
            a = a[a.index < cutoff]
        la = np.log(a.clip(lower=5))
        dw = daily_weather(weather)
        self.clim = DailyClim().fit(la, dw[dw.index < (cutoff or dw.index.max())])
        state = self._state(la, loo=True)
        origins = state.index[state["an7"].notna()]
        X = self._rows(origins, state, dw, loo=True)
        T = X.index.get_level_values("origin") + pd.to_timedelta(X.index.get_level_values("d"), unit="D")
        y = la.reindex(T).values - X["clim_T"].values
        ok = ~np.isnan(y)
        X, y, T = X[ok], y[ok], T[ok]
        w = 1.0 + np.isin(T.month, list(COLD_MONTHS))
        self.features = list(X.columns)
        self.booster = _gbm(DAILY_PARAMS, self.rounds, self.seed, X, y, w)
        return self

    def predict_frame(self, origins: pd.DatetimeIndex, pm: pd.Series,
                      dw: pd.DataFrame | None) -> pd.DataFrame:
        a = daily_aqi(pm)
        la = np.log(a.clip(lower=5))
        state = self._state(la)
        X = self._rows(origins, state, dw)
        z = self.booster.predict(X[self.features].values)
        out = pd.DataFrame(index=X.index)
        out["pred_aqi"] = np.exp(z + X["clim_T"].values)
        out["clim_aqi"] = np.exp(X["clim_T"].values)
        out["persist7_aqi"] = np.exp(X["lvl7"].values)
        # climatology shifted by the recent 30-day anomaly, decaying with horizon
        tau = 20.0
        out["anom_decay_aqi"] = np.exp(X["clim_T"].values + X["an30"].values *
                                       np.exp(-X.index.get_level_values("d").values / tau))
        return out


class LongRange:
    """Days 1..90: climatology of daily AQI, shifted by the recent anomaly which decays
    with e-folding time `tau` days. In backtests nothing beat this beyond the first week
    (the gradient-boosted daily model and weather-driven variants did not)."""

    def __init__(self, max_d: int = 90, tau: float = 10.0, k: float = 1.0):
        self.max_d, self.tau, self.k = max_d, tau, k

    def fit(self, pm: pd.Series, cutoff: pd.Timestamp | None = None) -> "LongRange":
        a = daily_aqi(pm)
        if cutoff is not None:
            a = a[a.index < cutoff]
        self.clim = SeasonalMean().fit(np.log(a.clip(lower=5)))
        # spread of daily log-AQI around climatology, by calendar month (for bands)
        la = np.log(a.clip(lower=5)).dropna()
        r = la - self.clim.at(la.index, season_of(la.index))
        self.q = r.groupby(r.index.month).quantile([0.1, 0.9]).unstack()
        # full spread by month (99 quantiles): day-to-day weather makes a single "expected"
        # value hide how often a date turns red; these give the chance per category
        qs = np.linspace(0.01, 0.99, 99)
        self.rq = {m: np.quantile(g.values, qs) for m, g in r.groupby(r.index.month)}
        # how often each calendar date was red / purple in past years (±15-day window)
        ad = a.dropna()
        self.freq = {thr: SeasonalMean(win=15).fit((ad > thr + 0.5).astype(float)) for thr in (150, 200)}
        return self

    def climate_prob(self, T: pd.DatetimeIndex, threshold: int) -> np.ndarray:
        """Share of past days around this date with daily AQI above `threshold`."""
        return self.freq[threshold].at(pd.DatetimeIndex(T))

    def prob_above(self, pred_aqi: np.ndarray, months: np.ndarray, threshold: float) -> np.ndarray:
        """P(daily mean AQI > threshold) given the point forecast, from the month's spread."""
        out = np.full(len(pred_aqi), np.nan)
        need = np.log(threshold) - np.log(np.asarray(pred_aqi, float))
        for m, q in self.rq.items():
            sel = months == m
            out[sel] = (q[None, :] > need[sel, None]).mean(axis=1)
        return out

    def predict_frame(self, origins: pd.DatetimeIndex, pm: pd.Series) -> pd.DataFrame:
        a = daily_aqi(pm)
        la = np.log(a.clip(lower=5))
        an = la - self.clim.at(la.index)
        an30 = an.rolling(30, min_periods=15).mean()
        D = np.arange(1, self.max_d + 1)
        o = pd.DatetimeIndex(np.repeat(origins.values, len(D)))
        d = np.tile(D, len(origins))
        T = o + pd.to_timedelta(d, unit="D")
        lc = self.clim.at(T)
        shift = self.k * np.nan_to_num(an30.reindex(o).values) * np.exp(-d / self.tau)
        out = pd.DataFrame(index=pd.MultiIndex.from_arrays([o, d], names=["origin", "d"]))
        out["pred_aqi"] = np.exp(lc + shift)
        out["clim_aqi"] = np.exp(lc)
        out["clim_lo"] = np.exp(lc + self.q.loc[T.month, 0.1].values)
        out["clim_hi"] = np.exp(lc + self.q.loc[T.month, 0.9].values)
        return out


def compose_daily(df: pd.DataFrame, hourly_days: int = 7, tau: float = 4.0) -> np.ndarray:
    """The 90-day product from columns origin, d, longrange_aqi, clim_aqi, hourly_aqi.

    Days 1..hourly_days: daily means of the hourly model. Afterwards: climatology plus the
    hourly model's day-7 anomaly, decaying with e-folding `tau` days, so the curve continues
    from where the weather-driven week ends (falls back to LongRange without it)."""
    lc = np.log(df["clim_aqi"].values)
    last = df[df["d"] == hourly_days].set_index("origin")
    an = (np.log(last["hourly_aqi"]) - np.log(last["clim_aqi"])).reindex(df["origin"]).values
    d = df["d"].values
    tail = np.exp(lc + an * np.exp(-(d - hourly_days) / tau))
    out = np.where(np.isnan(an), df["longrange_aqi"].values, tail)
    use_h = (d <= hourly_days) & ~np.isnan(df["hourly_aqi"].values)
    return np.where(use_h, df["hourly_aqi"].values, out)


class HorizonRouter:
    """Separate model variants per horizon group, each group served by the variant that
    did best on a held-out validation winter (see aqf.experiments / select_per_horizon)."""

    def __init__(self, variants: dict[str, HourlyModel], choice: dict[tuple[int, int], str]):
        self.variants = variants
        self.choice = choice  # (lo, hi) horizon group -> variant name

    def fit(self, pm, src, wfeat, cutoff=None) -> "HorizonRouter":
        for m in self.variants.values():
            m.fit(pm, src, wfeat, cutoff)
        self.n_train = sum(m.n_train for m in self.variants.values())
        return self

    def set_cams(self, cams: pd.DataFrame) -> None:
        """Fresh CAMS (history + forecast) for variants that use it, before predicting."""
        for m in self.variants.values():
            if getattr(m, "cams", None) is not None:
                m.cams = cams

    def predict_frame(self, origins, pm, wfeat, horizons=None) -> pd.DataFrame:
        used = sorted(set(self.choice.values()))
        frames = {v: self.variants[v].predict_frame(origins, pm, wfeat, horizons) for v in used}
        out = frames[used[0]].copy()  # baselines (persistence, climatology) are identical
        h = out.index.get_level_values("h").values
        for (lo, hi), v in self.choice.items():
            m = (h >= lo) & (h <= hi)
            out.loc[m, ["pred_lp", "pred_pm", "pred_aqi"]] = frames[v].loc[m, ["pred_lp", "pred_pm", "pred_aqi"]].values
        out["variant"] = ""
        for (lo, hi), v in self.choice.items():
            out.loc[(h >= lo) & (h <= hi), "variant"] = v
        return out


# Variants that won at least one horizon group on the 2024/25 selection winter
# (aqf.select_horizon); all use the four-model weather ensemble and inversion features.
HOURLY_VARIANTS = {
    "pred_v3_mm_l1": dict(lgbm_extra=dict(objective="l1")),            # median: 1–6 h
    "pred_v3_mm_cams": dict(use_cams=True),                             # + CAMS: 7–24 h
    "pred_v3_mm_big": dict(rounds=900, lgbm_extra=dict(learning_rate=0.02, num_leaves=63,
                                                        min_child_samples=300)),  # 1–7 days
    "pred_v2_mm": dict(),
    # evolutionary search (aqf.tune), kept only where it beat the current model on both
    # unseen winters (2025/26 and Oct 2026)
    "tuned_4_6": dict(rounds=476, lgbm_extra=dict(learning_rate=0.064, num_leaves=36,
                      min_child_samples=362, colsample_bytree=0.3, subsample=1.0,
                      reg_lambda=5.14, objective="l1")),
    "tuned_37_48": dict(rounds=1200, lgbm_extra=dict(learning_rate=0.077, num_leaves=21,
                        min_child_samples=181, colsample_bytree=0.514, subsample=0.759,
                        reg_lambda=1.128, objective="l1")),
}


def make_hourly_model(choice: dict[str, str], cams: pd.DataFrame | None) -> HorizonRouter:
    """One model per horizon group, as chosen in horizon_choice.json ("lo-hi" -> variant)."""
    by_variant: dict[str, list[tuple[int, int]]] = {}
    for key, v in choice.items():
        lo, hi = map(int, key.split("-"))
        by_variant.setdefault(v, []).append((lo, hi))
    variants = {}
    for v, groups in by_variant.items():
        kw = dict(HOURLY_VARIANTS[v])
        if kw.pop("use_cams", False):
            kw["cams"] = cams
        variants[v] = HourlyModel(max_h=168, only_groups=groups, **kw)
    route = {(lo, hi): v for v, gs in by_variant.items() for lo, hi in gs}
    return HorizonRouter(variants, route)
