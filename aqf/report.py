"""Summarise backtests: metric tables, error bands for the live forecast, dashboard data.

    .venv/bin/python -m aqf.report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import aqi
from .backtest import season_bounds
from .data import PROC, load_target
from .models import GBM_VERSION, LongRange, compose_daily, daily_aqi

INTERVALS = PROC / "intervals.json"
SUMMARY = PROC / "backtest_summary.json"

H_TABLE = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 48), (49, 72), (73, 120), (121, 168)]
D_TABLE = [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 90)]
HOURLY_DAYS = 7  # the 90-day product uses the hourly model for its first week
RECENT_FROM = 2022  # the live error bands come from these seasons on
RED_FLAG = 0.3  # the week tiles mark a day whose red-day chance is at least this


def realistic(df: pd.DataFrame) -> pd.DataFrame:
    """Rows scored with archived weather forecasts where they exist, reanalysis otherwise
    (decided per retraining period, since the forecast archive starts in 2024-01)."""
    key = ["season", "trained_until"] if "trained_until" in df else ["season"]
    has_fc = df.loc[df["weather"] == "forecast", key].drop_duplicates()
    has_fc["_fc"] = True
    d = df.merge(has_fc, on=key, how="left")
    keep = (d["weather"] == "forecast") | ((d["weather"] == "actual") & d["_fc"].isna())
    return d[keep].drop(columns="_fc")


def mae(p, a) -> float:
    p, a = np.asarray(p, float), np.asarray(a, float)
    ok = ~np.isnan(p) & ~np.isnan(a)
    return float(np.abs(p[ok] - a[ok]).mean())


def log_resid(pred_aqi, act_aqi) -> np.ndarray:
    return np.log1p(aqi.aqi_to_pm(act_aqi)) - np.log1p(aqi.aqi_to_pm(pred_aqi))


def bands(key: np.ndarray, r: np.ndarray, groups) -> list[dict]:
    out = []
    for lo, hi in groups:
        m = (key >= lo) & (key <= hi) & ~np.isnan(r)
        if m.sum() >= 50:
            out.append({"lo": lo, "hi": hi, "q10": float(np.quantile(r[m], 0.1)),
                        "q90": float(np.quantile(r[m], 0.9)), "n": int(m.sum())})
    return out


def hourly_table(df: pd.DataFrame) -> list[dict]:
    rows = []
    for lo, hi in H_TABLE:
        g = df[(df["h"] >= lo) & (df["h"] <= hi)]
        if g.empty:
            continue
        a = g["actual_aqi"]
        rows.append({
            "b": f"{lo}–{hi}", "n": int(len(g)),
            "model": mae(g["pred_aqi"], a), "persist": mae(g["persist_aqi"], a),
            "same_hour": mae(g["same_hour_aqi"], a), "clim": mae(g["clim_aqi"], a),
            "bias": float((g["pred_aqi"] - a).mean()),
            "cat": float((aqi.category_index(g["pred_aqi"]) == aqi.category_index(a)).mean()),
            "cat_persist": float((aqi.category_index(g["persist_aqi"]) == aqi.category_index(a)).mean()),
        })
    return rows


def hourly_to_daily(hb: pd.DataFrame) -> pd.DataFrame:
    """Daily means from hourly forecasts issued at midnight (origin = 00:00 of day X).

    Day X keeps its observed 00:00 value plus 23 forecast hours; X..X+6 map to d=1..7
    with the daily convention origin = last complete day (X-1)."""
    g = hb[hb["origin"].dt.hour == 0].copy()
    g["day"] = g["T"].dt.normalize()
    g = g[g["T"].dt.hour != 0]  # 00:00 of the next day belongs to the next day; drop it
    obs0 = g.groupby("origin")["persist_aqi"].first()
    agg = g.groupby(["origin", "day"])["pred_aqi"].agg(["sum", "count"]).reset_index()
    first = agg["day"] == agg["origin"]
    agg.loc[first, "sum"] += obs0.reindex(agg.loc[first, "origin"]).values
    agg.loc[first, "count"] += 1
    agg = agg[agg["count"] >= 20]
    agg["pred_aqi"] = agg["sum"] / agg["count"]
    agg["d"] = (agg["day"] - agg["origin"]).dt.days + 1
    agg["origin"] = agg["origin"] - pd.Timedelta(days=1)
    return agg[["origin", "d", "pred_aqi"]]


def daily_product_backtest(hb: pd.DataFrame) -> pd.DataFrame:
    """Score the 90-day product: hourly model for days 1–7, LongRange afterwards. LongRange
    is refit at every month start, like the hourly model in the backtest."""
    hd = hourly_to_daily(hb)
    res = []
    for y in sorted(hb["season"].unique()):
        start, end = season_bounds(y)
        cutoffs = pd.date_range(start, end, freq="MS", inclusive="left")
        for i, cut in enumerate(cutoffs):
            nxt = cutoffs[i + 1] if i + 1 < len(cutoffs) else end
            pm = load_target(profile_cutoff=cut)[0]["pm"]
            a = daily_aqi(pm)
            lr = LongRange().fit(pm, cutoff=cut)
            # origin = last complete day; the first origin of a month is the day before it
            origins = pd.date_range(cut - pd.Timedelta(days=1), nxt - pd.Timedelta(days=2), freq="D")
            origins = origins[origins < a.dropna().index.max()]
            if not len(origins):
                continue
            p = lr.predict_frame(origins, pm).reset_index()
            p = p.merge(hd.rename(columns={"pred_aqi": "hourly_aqi"}), on=["origin", "d"], how="left")
            p["T"] = p["origin"] + pd.to_timedelta(p["d"], unit="D")
            p["actual_aqi"] = a.reindex(p["T"]).values
            la = np.log(a.clip(lower=5))
            p["persist7_aqi"] = np.exp(la.rolling(7, min_periods=4).mean()).reindex(p["origin"]).values
            p["longrange_aqi"] = p["pred_aqi"]
            p["product_aqi"] = compose_daily(p, HOURLY_DAYS)
            for thr, col in ((150, "p_red"), (200, "p_purple")):
                p[col] = lr.prob_above(p["product_aqi"].values, p["T"].dt.month.values, thr)
            p["season"] = y
            res.append(p.dropna(subset=["actual_aqi"]))
    return pd.concat(res, ignore_index=True)


def red_calibration(dp: pd.DataFrame) -> dict:
    """Is "30% chance of a red day" right 30% of the time? Days 8–90 of the 90-day product."""
    g = dp[(dp["d"] >= 8) & dp["p_red"].notna()]
    red = (g["actual_aqi"] > 150.5).astype(float)
    bins = pd.cut(g["p_red"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], include_lowest=True)
    tab = pd.DataFrame({"pred": g["p_red"], "obs": red}).groupby(bins, observed=True).agg(
        pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size"))
    base = red.groupby(g["T"].dt.month).transform("mean")  # in-sample monthly base rate
    brier, brier_clim = ((g["p_red"] - red) ** 2).mean(), ((base - red) ** 2).mean()
    return {"bins": [{"bin": str(i), **{k: float(v) for k, v in r.items()}} for i, r in tab.iterrows()],
            "brier": float(brier), "brier_monthly_base_rate": float(brier_clim)}


def red_week(dp: pd.DataFrame | None = None) -> dict:
    """How the week tiles' red-day mark did on past winters (days 1–7 of the seasons the
    daily_q bands come from): how many of the marked days were red, and how many red days
    were marked, by day of the forecast."""
    dp = pd.read_parquet(PROC / "bt_daily_product.parquet") if dp is None else dp
    g = dp[(dp["season"] >= RECENT_FROM) & (dp["d"] <= HOURLY_DAYS)].dropna(subset=["product_aqi", "actual_aqi"])
    p = np.full(len(g), np.nan)
    for b in json.loads(INTERVALS.read_text())["daily_q"]:  # as aqf.forecast computes p_red
        m = ((g["d"] >= b["lo"]) & (g["d"] <= b["hi"])).values
        p[m] = (np.array(b["q"])[None, :] > (np.log(150) - np.log(g["product_aqi"].values[m]))[:, None]).mean(axis=1)
    red, marked, d = (g["actual_aqi"] > 150.5).values, p >= RED_FLAG, g["d"].values
    return {"flag": RED_FLAG, "from": RECENT_FROM, "red_when_marked": round(float(red[marked].mean()), 2),
            "marked_by_day": [round(float(marked[red & (d == k)].mean()), 2) for k in range(1, HOURLY_DAYS + 1)]}


def daily_table(dp: pd.DataFrame) -> list[dict]:
    rows = []
    for lo, hi in D_TABLE:
        g = dp[(dp["d"] >= lo) & (dp["d"] <= hi)]
        a = g["actual_aqi"]
        rows.append({"b": f"{lo}–{hi}" if hi > lo else str(lo), "n": int(len(g)),
                     "model": mae(g["product_aqi"], a), "longrange": mae(g["longrange_aqi"], a),
                     "hourly": mae(g["hourly_aqi"], a) if g["hourly_aqi"].notna().any() else None,
                     "persist": mae(g["persist7_aqi"], a), "clim": mae(g["clim_aqi"], a)})
    return rows


SRC_LABEL = {2020: "US Embassy", 2021: "US Embassy", 2022: "US Embassy",
             2023: "US Embassy", 2024: "US Embassy → IQAir", 2025: "IQAir", 2026: "IQAir"}


def season_rows(hr: pd.DataFrame) -> list[dict]:
    out = []
    for y, g in hr.groupby("season"):
        s = g[g["h"] <= 24]
        l = g[g["h"] >= 49]
        label = f"{y}/{str(y + 1)[2:]}" + (" (Oct 1–17)" if y == 2026 else "")
        out.append({"season": label, "src": SRC_LABEL.get(y, ""), "y": int(y),
                    "src_code": "emb" if y <= 2023 else ("mix" if y == 2024 else "iq"),
                    "partial": bool(g["T"].max() < season_bounds(y)[1] - pd.Timedelta(days=30)),
                    "n": int(g["origin"].nunique()),
                    "h24": mae(s["pred_aqi"], s["actual_aqi"]), "h24p": mae(s["persist_aqi"], s["actual_aqi"]),
                    "d37": mae(l["pred_aqi"], l["actual_aqi"]), "d37c": mae(l["clim_aqi"], l["actual_aqi"]),
                    "weather": "forecast" if (g["weather"] == "forecast").any() else "actual"})
    return out


def summarize() -> dict:
    hb = pd.read_parquet(PROC / "bt_hourly.parquet")
    hr = realistic(hb)
    dp = daily_product_backtest(hr)
    dp.to_parquet(PROC / "bt_daily_product.parquet")

    def err(df, pred, act="actual_aqi"):
        e = (df[pred] - df[act]).abs()
        return {"mae": float(e.mean()), "mape": float((e / df[act]).mean() * 100)}

    last = hr[hr["season"] == 2025]  # "last winter", Oct 2025 – Mar 2026
    dl = dp[dp["season"] == 2025]
    out = {
        "headline": {
            "h3": err(last[last["h"] <= 3], "pred_aqi"),
            "h24": err(last[last["h"] <= 24], "pred_aqi"),
            "h168": err(last, "pred_aqi"),
            "d17": err(dl[dl["d"] <= 7], "product_aqi"),
            "d890": err(dl[dl["d"] > 7], "product_aqi"),
        },
        "hourly": {"iqair": hourly_table(hr[hr["season"] >= 2025]), "all": hourly_table(hr)},
        "hourly_by_weather": {
            w: hourly_table(hb[(hb["weather"] == w) & (hb["season"] >= 2024)]) for w in ("actual", "forecast")
        },
        "daily": daily_table(dp),
        "daily_iqair": daily_table(dp[dp["season"] >= 2025]),
        "seasons": season_rows(hr),
        "n_forecasts": int(hr["origin"].nunique()),
        "model_version": GBM_VERSION,  # the LightGBM models' version; the site says which version these numbers are for
    }
    out["red_calibration"] = red_calibration(dp)
    SUMMARY.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    recent = hr[hr["season"] >= RECENT_FROM]
    dpr = dp[dp["season"] >= RECENT_FROM]
    INTERVALS.write_text(json.dumps({
        "hourly": bands(recent["h"].values, log_resid(recent["pred_aqi"], recent["actual_aqi"]),
                        [(h, h) for h in range(1, 7)] + [(7, 12), (13, 24), (25, 48), (49, 72),
                                                         (73, 96), (97, 120), (121, 144), (145, 168)]),
        "daily": bands(dpr["d"].values, log_resid(dpr["product_aqi"], dpr["actual_aqi"]),
                       [(1, 1), (2, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 90)]),
        # log(actual daily AQI) - log(forecast), 99 quantiles, for P(red) on days 1-7
        "daily_q": [{"lo": lo, "hi": hi, "q": np.quantile(
            (np.log(dpr["actual_aqi"]) - np.log(dpr["product_aqi"]))[(dpr["d"] >= lo) & (dpr["d"] <= hi)].dropna(),
            np.linspace(0.01, 0.99, 99)).tolist()} for lo, hi in ((1, 1), (2, 3), (4, 7))],
    }, indent=1))
    return out


def _print(rows: list[dict], cols: list[str], title: str) -> None:
    print(f"\n{title}")
    print(pd.DataFrame(rows).set_index("b")[cols].round(2).to_string())


if __name__ == "__main__":
    o = summarize()
    _print(o["hourly"]["iqair"], ["n", "model", "persist", "same_hour", "clim", "bias", "cat", "cat_persist"],
           "HOURLY — IQAir seasons (2025/26 + Oct 2026), realistic weather")
    _print(o["hourly"]["all"], ["n", "model", "persist", "same_hour", "clim", "bias", "cat", "cat_persist"],
           "HOURLY — all seasons 2020–2026, realistic weather")
    _print(o["hourly_by_weather"]["actual"], ["model", "persist", "clim"], "HOURLY 2024+ with actual weather")
    _print(o["hourly_by_weather"]["forecast"], ["model", "persist", "clim"], "HOURLY 2024+ with forecast weather")
    _print(o["daily"], ["n", "model", "hourly", "longrange", "persist", "clim"], "DAILY product — all seasons")
    _print(o["daily_iqair"], ["n", "model", "hourly", "longrange", "persist", "clim"], "DAILY product — IQAir seasons")
    print("\nSEASONS"); print(pd.DataFrame(o["seasons"]).round(1).to_string(index=False))
