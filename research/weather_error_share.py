"""How much of the forecast error comes from errors of the weather forecast? (owner, 2026-11-06)

The backtest (aqf.backtest, walk-forward, monthly retraining) issues every forecast from 2024-02 on
twice with the same model: with the weather forecasts archived as issued (what the site really
had), and with the weather that then happened (Open-Meteo reanalysis). The difference between the
two is what the errors of the weather forecast cost; what is left with the actual weather is
everything else (how the pollution itself changed, smoke and dust from outside, what the model
does not capture, measurement). Nothing is trained or written to the models; reads
data/processed/bt_hourly.parquet and bt_daily_product.parquet (as recomputed 2026-11-05).

Measures, per horizon and per winter:
  mae_fc, mae_act  — MAE with the forecast and with the actual weather
  weather          — mae_fc - mae_act: how much smaller the error would be with a perfect weather
                     forecast (95% bootstrap interval over forecast days)
  share            — weather / mae_fc
  shift            — mean |forecast(fc weather) - forecast(actual weather)|: how far the weather
                     errors move the forecast at all
  sq_weather etc.  — squared error split: (F-A)² = (F-P)² + (P-A)² + 2(F-P)(P-A), F with the
                     forecast weather, P with the actual weather, A measured; shares of mean (F-A)²

The daily product is built as on the site: days 1-7 = daily means of the hourly model from
00:00 forecasts (aqf.report.hourly_to_daily), later days = climatology + the day-7 anomaly
(aqf.models.compose_daily).

    PYTHONPATH=. .venv/bin/python research/weather_error_share.py

    PYTHONPATH=. .venv/bin/python research/weather_error_share.py --perfect  # ~17 min, 16 retrainings

Writes data/processed/weather_error_share.json (and weather_error_share_perfect.parquet).

Result (2026-11-06). "perfect" = model trained on and given the actual weather (reanalysis);
"airport" = the same with the wind measured at Tashkent airport (METAR) in place of the reanalysis
wind ("airport+" also its temperature and humidity: no better than the wind alone). MAE drop against
the forecast with the weather forecasts as archived (95% interval over days).

  The site's own leads (06:00 issue, daily mean of the day 1 / 3 days later; 434 / 423 days 2024-02 .. 2026-11):
                 now     perfect                  airport
  1 day ahead    20.2    18.7  -1.5 (0.6..2.5) 8%    17.6  -2.6 (1.7..3.6) 13%
  3 days ahead   23.4    20.4  -2.9 (1.6..4.2) 12%   18.8  -4.5 (3.2..5.8) 19%
  28.10-3.11.2026 (7 days): 1 day 25.1 -> 21.8 -> 18.3 (-27%); 3 days 30.3 -> 22.3 -> 19.0 (-37%; airport+ 15.4, -49%)
  October 2026: 3 days 27.5 -> 21.4 -> 19.7 (-29%)

  The daily product by horizon (d = 1 is the issue day itself from 00:00, 1-23 h ahead):
  days    now     same model, act. wx   perfect        airport
  1       15.0    15.4                  15.0   0%      14.7   2%
  2-3     22.3    21.1                  20.2   9%      19.3   14%
  4-7     25.4    22.4                  21.3   16%     20.0   21%
  8-14    33.0    32.8                  32.8   1%      33.1   -
  15-90   climatology, weather does not enter: 0%
Hourly: 1-24 h 0-2%, 25-48 h 4%, 49-72 h 7%, 73-120 h 9%, 121-168 h 14% (perfect).
The weather errors move the forecast a lot (mean |shift| 5 / 10 / 13 AQI on days 1 / 2-3 / 4-7;
on 16-21% of days of 2-7 days ahead by more than 20 AQI, and then perfect weather would cut the
error by ~30%), but on most days the weather forecast is close enough, and the shift goes towards
the truth about as often as away from it; the reanalysis is itself a model estimate (grid of ~25 km),
and its own errors appear in the weather part with one sign and in the rest with the other. The
wind measured in town is worth 5-7 points more than the reanalysis wind. So a perfect weather
forecast would take 13% off the error at 1 day and 19% at 3 days (a third and more in stagnant
spells like late October 2026); the rest (17-19 AQI) is not the weather. A real forecast would get
only part of it. Oct 2026: even with perfect weather the forecast is too high (bias +6..+12): the
drift is a level shift. The site's "Weather" column (aqf/attribution.py) is the shift, not the cost.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqf.backtest import season_bounds
from aqf.data import PROC, RAW, load_target, load_weather
from aqf.features import weather_features
from aqf.models import compose_daily, make_hourly_model
from aqf.report import D_TABLE, H_TABLE, HOURLY_DAYS, hourly_to_daily

PERIODS = {"2023/24 (Feb-Mar)": 2023, "2024/25": 2024, "2025/26": 2025, "Oct 2026": 2026}
PERFECT = PROC / "weather_error_share_perfect.parquet"


METAR = RAW / "weather" / "metar_UZTT_full.csv"  # Tashkent airport, IEM, local time, 2017-12 .. 2026-11-04
STATION = ("wind", "all")  # station weather in place of the reanalysis: 10 m wind; wind + temperature and humidity


def station_wx(kind: str) -> pd.DataFrame:
    """The reanalysis with the airport's measured 10 m wind (and, for "all", 2 m temperature and humidity)
    put in its place: what a perfect forecast of the weather measured in town would give. Hourly means
    of the half-hourly METAR (direction as the vector mean); the 0.2% of hours without a report take
    the reanalysis (wind scaled by the mean airport / reanalysis ratio)."""
    wx = load_weather()
    m = pd.read_csv(METAR, na_values="M", usecols=["valid", "sknt", "drct", "tmpc", "relh"])
    m.index = pd.to_datetime(m.pop("valid"))
    ws = m["sknt"] * 1.852
    rad = np.deg2rad(m["drct"])
    h = pd.DataFrame({"ws": ws, "u": -ws * np.sin(rad), "v": -ws * np.cos(rad), "t": m["tmpc"], "rh": m["relh"]})
    h = h.resample("h").mean().reindex(wx.index)
    era = wx["wind_speed_10m"].astype(float)
    ok = h["ws"].notna() & era.notna()
    ratio = h.loc[ok, "ws"].mean() / era[ok].mean()
    wx = wx.astype(float)
    wx["wind_speed_10m"] = h["ws"].fillna(era * ratio)
    wd = np.rad2deg(np.arctan2(-h["u"], -h["v"])) % 360
    wx["wind_direction_10m"] = wd.where(h["ws"] > 0).fillna(wx["wind_direction_10m"])
    if kind == "all":
        wx["temperature_2m"] = h["t"].fillna(wx["temperature_2m"])
        wx["relative_humidity_2m"] = h["rh"].fillna(wx["relative_humidity_2m"])
    print(f"station weather '{kind}': {ok.mean():.3f} of hours measured, wind ratio {ratio:.2f}", flush=True)
    return wx


def perfect_run(check: bool = False, station: str | None = None) -> pd.DataFrame:
    """The backtest of aqf.backtest.backtest_hourly for the retrainings that have archived forecast
    weather, but with a model trained on the actual weather (reanalysis, as before 2024) and given
    the actual weather: the forecast in a world with perfect weather forecasts. The served model is
    trained on the archived forecasts of the matching lead and has learnt to trust far-ahead
    weather less, so giving it the actual weather (the backtest's "actual" variant) understates
    what perfect weather would be worth. `check`: train as the backtest does (forecast weather)
    on the first retraining and compare with its "actual" rows (same protocol)."""
    wf = weather_features(station_wx(station) if station else load_weather())
    choice = json.loads((PROC / "horizon_choice.json").read_text())
    cams = pd.read_parquet(RAW / "weather" / "cams.parquet")
    bt = pd.read_parquet(PROC / "bt_hourly.parquet", columns=["origin", "h", "pred_aqi", "weather", "trained_until"])
    cuts = sorted(bt.loc[bt["weather"] == "forecast", "trained_until"].unique())
    res = []
    for cut in map(pd.Timestamp, cuts):
        y = cut.year if cut.month >= 10 else cut.year - 1
        nxt = min(cut + pd.offsets.MonthBegin(1), season_bounds(y)[1])
        tgt, _ = load_target(profile_cutoff=cut)
        pm, src = tgt["pm"], tgt["src"]
        origins = pd.date_range(cut, min(nxt, pm.dropna().index.max()), freq="3h", inclusive="left")
        origins = origins[np.log1p(pm).ffill(limit=2).reindex(origins).notna().values]
        if check:
            from aqf.data import load_prev_runs_ensemble, splice_forecast_weather
            m = make_hourly_model(choice, cams).fit(pm, src, splice_forecast_weather(load_weather(), load_prev_runs_ensemble()), cutoff=cut)
            p = m.predict_frame(origins, pm, wf).reset_index()[["origin", "h", "pred_aqi"]]
            ref = bt[(bt["weather"] == "actual") & (bt["trained_until"] == cut)]
            x = p.merge(ref, on=["origin", "h"], suffixes=("", "_bt"))
            print(f"check {cut:%Y-%m}: {len(x)} rows, max |diff| {np.abs(x['pred_aqi'] - x['pred_aqi_bt']).max():.4f} AQI")
            return x
        t0 = pd.Timestamp.now()
        m = make_hourly_model(choice, cams).fit(pm, src, wf, cutoff=cut)
        p = m.predict_frame(origins, pm, wf).reset_index()[["origin", "h", "pred_aqi"]]
        p["trained_until"] = cut
        res.append(p)
        print(f"  {cut:%Y-%m}: {len(origins)} origins, {(pd.Timestamp.now() - t0).total_seconds():.0f}s", flush=True)
    out = pd.concat(res, ignore_index=True)
    out.to_parquet(PROC / f"weather_error_share_station_{station}.parquet" if station else PERFECT)
    return out


def boot_diff(err_f: pd.Series, err_a: pd.Series, day: pd.Series, n: int = 2000) -> tuple[float, float]:
    """95% interval of mean(err_f - err_a), resampling forecast days."""
    e = (err_f - err_a).groupby(day.values)
    s, c = e.sum().values, e.count().values
    idx = np.random.default_rng(0).integers(0, len(s), size=(n, len(s)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def split(F: np.ndarray, P: np.ndarray, A: np.ndarray, day: pd.Series) -> dict:
    ok = ~np.isnan(F) & ~np.isnan(P) & ~np.isnan(A)
    F, P, A, day = F[ok], P[ok], A[ok], day[ok]
    ef, ea = np.abs(F - A), np.abs(P - A)
    w, r = F - P, P - A
    mse = np.mean((F - A) ** 2)
    lo, hi = boot_diff(pd.Series(ef), pd.Series(ea), day)
    return {"n": int(ok.sum()), "mae_fc": round(float(ef.mean()), 2), "mae_act": round(float(ea.mean()), 2),
            "weather": round(float(ef.mean() - ea.mean()), 2), "ci": [round(lo, 2), round(hi, 2)],
            "share": round(float((ef.mean() - ea.mean()) / ef.mean()), 3),
            "shift": round(float(np.abs(w).mean()), 2),
            "bias_fc": round(float((F - A).mean()), 2), "bias_act": round(float((P - A).mean()), 2),
            "sq_weather": round(float(np.mean(w ** 2) / mse), 3), "sq_rest": round(float(np.mean(r ** 2) / mse), 3),
            "sq_cross": round(float(2 * np.mean(w * r) / mse), 3)}


def paired_hourly() -> pd.DataFrame:
    b = pd.read_parquet(PROC / "bt_hourly.parquet", columns=["origin", "h", "T", "pred_aqi", "actual_aqi", "clim_aqi",
                                                             "persist_aqi", "season", "weather"])
    f, a = b[b["weather"] == "forecast"], b[b["weather"] == "actual"]
    p = f.merge(a[["origin", "h", "pred_aqi"]], on=["origin", "h"], suffixes=("_fc", "_act"))
    assert len(p) == len(f), "every forecast-weather row has its actual-weather twin"
    for col, path in RUNS.items():
        q = pd.read_parquet(path, columns=["origin", "h", "pred_aqi"]).rename(columns={"pred_aqi": f"pred_aqi_{col}"})
        p = p.merge(q, on=["origin", "h"], how="left")
        assert p[f"pred_aqi_{col}"].notna().mean() > 0.999, f"the {col} run covers the backtest's forecast rows"
    return p


def paired_daily(p: pd.DataFrame) -> pd.DataFrame:
    dp = pd.read_parquet(PROC / "bt_daily_product.parquet")
    out = dp[dp["origin"] >= p["origin"].min() - pd.Timedelta(days=1)].copy()
    for v in ("fc", "act", *RUNS):
        hd = hourly_to_daily(p.rename(columns={f"pred_aqi_{v}": "pred_aqi"})).rename(columns={"pred_aqi": "hourly_aqi"})
        x = out.drop(columns="hourly_aqi").merge(hd, on=["origin", "d"], how="left")
        out[f"prod_{v}"] = compose_daily(x, HOURLY_DAYS)
        out[f"has_{v}"] = x.groupby("origin")["hourly_aqi"].transform(lambda s: s.notna().sum() == HOURLY_DAYS).values
    out = out[out[[f"has_{v}" for v in ("fc", "act", *RUNS)]].all(axis=1)]
    # the forecast-weather product is the one the site's daily table scores for these winters
    same = np.abs(out["prod_fc"] - out["product_aqi"])
    assert np.nanmax(same) < 1e-6, f"rebuilt product differs from bt_daily_product by {np.nanmax(same)}"
    return out


VARIANTS = {"same_model": "act", "perfect": "perf"}  # what "actual weather" means: see the docstring
RUNS = {"perf": PERFECT, **{f"st_{k}": PROC / f"weather_error_share_station_{k}.parquet" for k in STATION
                            if (PROC / f"weather_error_share_station_{k}.parquet").exists()}}
WEEK = ("2026-10-28", "2026-11-03")  # the committee week (docs/checks/2026-10-27-committee-vs-forecast.md)


def station_summary(p: pd.DataFrame, d: pd.DataFrame) -> dict:
    """Perfect weather from the reanalysis against perfect weather with the airport's measured wind
    (and temperature, humidity): does a forecast of the wind as measured in town add to it?
    Daily product by horizon, and the site's own days (06:00 issue, the day 1 and 3 days later)."""
    cols = ["fc", "perf", *[c for c in RUNS if c.startswith("st_")]]
    out = {"daily": {}, "issue06": {}}
    for name, y in {"all": None, **PERIODS}.items():
        dd = d if y is None else d[d["season"] == y]
        out["daily"][name] = {}
        for lo, hi in D_TABLE[:4]:
            g = dd[dd["d"].between(lo, hi)]
            r = {c: round(float(np.abs(g[f"prod_{c}"] - g["actual_aqi"]).mean()), 2) for c in cols}
            for c in cols[2:]:
                ci = boot_diff(np.abs(g["prod_perf"] - g["actual_aqi"]), np.abs(g[f"prod_{c}"] - g["actual_aqi"]), g["origin"])
                r[f"{c}_vs_perf"] = [round(r["perf"] - r[c], 2), *[round(x, 2) for x in ci]]
            out["daily"][name][str(lo) if lo == hi else f"{lo}-{hi}"] = r
    p = p.assign(date=p["T"].dt.normalize())
    for lead in (1, 3):
        s = p[(p["origin"].dt.hour == 6) & ((p["date"] - p["origin"].dt.normalize()).dt.days == lead)]
        g = s.groupby("date").agg(n=("h", "size"), act=("actual_aqi", "mean"),
                                  **{c: (f"pred_aqi_{c}", "mean") for c in cols})
        g = g[g["n"] >= 18]
        season = np.where(g.index.month >= 7, g.index.year, g.index.year - 1)
        for name, sel in (("week", (g.index >= WEEK[0]) & (g.index <= WEEK[1])), ("oct2026", g.index >= "2026-10-01"),
                          ("2024/25", season == 2024), ("2025/26", season == 2025), ("all", g.index == g.index)):
            x = g[sel]
            r = {"days": int(len(x)), **{c: round(float(np.abs(x[c] - x["act"]).mean()), 1) for c in cols},
                 **{f"bias_{c}": round(float((x[c] - x["act"]).mean()), 1) for c in cols}}
            for c in cols[1:]:  # MAE drop against the forecast as issued, 95% interval over days
                d0 = np.abs(x["fc"] - x["act"]) - np.abs(x[c] - x["act"])
                bs = d0.values[np.random.default_rng(0).integers(0, len(d0), size=(2000, len(d0)))].mean(1)
                r[f"gain_{c}"] = [round(float(d0.mean()), 1), round(float(np.percentile(bs, 2.5)), 1), round(float(np.percentile(bs, 97.5)), 1),
                                  f"{d0.mean() / np.abs(x['fc'] - x['act']).mean():.0%}"]
            out["issue06"][f"{name}_lead{lead}"] = r
    for k, t in out.items():
        for name, r in t.items():
            print(f"{k:8s} {name:22s} {r}")
    return out


def main() -> dict:
    p = paired_hourly()
    d = paired_daily(p)
    res = {v: {"hourly": {}, "daily": {}} for v in VARIANTS}
    for v, col in VARIANTS.items():
        for name, y in {"all": None, **PERIODS}.items():
            hp = p if y is None else p[p["season"] == y]
            dd = d if y is None else d[d["season"] == y]
            res[v]["hourly"][name] = {f"{lo}-{hi}": split(hp.loc[m, "pred_aqi_fc"].values, hp.loc[m, f"pred_aqi_{col}"].values,
                                                          hp.loc[m, "actual_aqi"].values, hp.loc[m, "origin"].dt.normalize())
                                      for lo, hi in H_TABLE + [(1, 168)] if (m := hp["h"].between(lo, hi)).any()}
            res[v]["daily"][name] = {}
            for lo, hi in D_TABLE:
                m = dd["d"].between(lo, hi)
                if m.any():
                    s = split(dd.loc[m, "prod_fc"].values, dd.loc[m, f"prod_{col}"].values, dd.loc[m, "actual_aqi"].values,
                              dd.loc[m, "origin"])
                    s["clim"] = round(float(np.abs(dd.loc[m, "clim_aqi"] - dd.loc[m, "actual_aqi"]).mean()), 2)
                    res[v]["daily"][name][str(lo) if lo == hi else f"{lo}-{hi}"] = s
            res[v]["daily"][name]["_days"] = f"{dd['T'].min():%Y-%m-%d} .. {dd['T'].max():%Y-%m-%d}"
    for kind in ("daily", "hourly"):
        for name in ("all", *PERIODS):
            a, b = res["same_model"][kind][name], res["perfect"][kind][name]
            rows = {g: {"n": b[g]["n"], "fc": b[g]["mae_fc"], "same_act": a[g]["mae_act"], "w_same": a[g]["weather"],
                        "perfect": b[g]["mae_act"], "w_perfect": b[g]["weather"], "ci": b[g]["ci"], "share": b[g]["share"],
                        "shift": b[g]["shift"], "bias_fc": b[g]["bias_fc"], "bias_perf": b[g]["bias_act"],
                        "sq_w": b[g]["sq_weather"], "sq_rest": b[g]["sq_rest"], "sq_x": b[g]["sq_cross"],
                        **({"clim": b[g]["clim"]} if kind == "daily" else {})} for g in b if g != "_days"}
            print(f"\n{kind.upper()} — {name} {b.get('_days', '')}")
            print(pd.DataFrame(rows).T.to_string())
    print("\nSTATION WEATHER vs REANALYSIS (perfect weather both)")
    res["station"] = station_summary(p, d)
    (PROC / "weather_error_share.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        perfect_run(check=True)
    elif "--perfect" in sys.argv:
        perfect_run()
    elif "--station" in sys.argv:
        perfect_run(station=sys.argv[sys.argv.index("--station") + 1])
    else:
        main()
