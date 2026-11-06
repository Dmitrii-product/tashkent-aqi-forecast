"""IQAir's note of 2026-10-26: was the rise dust, not the city's own pollution? And do westerly
winds bring dirty air to Tashkent? (owner's questions, 2026-10-28)

The note says: AQI above 150 at 13:00 on 26 October, PM10 dominant, an "acute autumn dust episode";
dust gave 52–78% of PM2.5 in late September – early October (Hydromet research institute, via
Gazeta); strong westerly winds bring dust from the Kyzylkum and Kazakhstan; very low NO2, CO, SO2
and O3 on 26 October show that traffic and industry are not the main drivers.

What we can check without accounts:
  * Sentinel-5P UV aerosol index (AER_AI, 340/380 nm) from the same public MEEO copy as our NO2:
    it rises above ~1 over UV-absorbing aerosol (dust, smoke) and is the satellite's dust signal.
    Read once a day (~13:00 local, qa ≥ 80) over the city (≤ 30 km), the region (≤ 300 km) and
    the upwind west (bearing 200–340°, 100–500 km), September–November 2018–2026, cached in
    data/research_cache/aer_ai/.
  * our Sentinel-5P NO2 over the city (burning: traffic, heating, power plants);
  * ERA5 wind (speed, gusts, direction), CAMS dust, the hourly city AQI and its day/night shape;
  * for the wind question: how PM2.5 changes after westerly wind, and whether the model's errors
    (backtest with archived forecast weather) depend on it.

    PYTHONPATH=. .venv/bin/python research/dust_claim_check.py          # ~15 min the first time

Result (2026-10-28):
  * the 52–78% (mean 64%) is a lab analysis of PM2.5 samples for 28 Sep – 6 Oct (47–52% on 13–14
    Oct; Ministry of Ecology briefing of 21 Oct), not for 24–27 Oct. Plausible then: before heating,
    and 20 Sep – 10 Oct 2026 had the cleanest air of any year (AQI 78), so dust was a large share
    of a small total.
  * 25–27 Oct: no strong wind (mean 4–5 km/h, gusts ≤ 23, from the west ≤ 23), peaks at night under
    calm easterly drainage (night/afternoon AQI 1.12–1.22, within the usual October 1.09–1.44),
    NO2 above the October median on 23, 24, 26 Oct (76–93rd percentile; the note says "very low"),
    CAMS dust ≈ 0. Our model, which sees no dust, forecast 157 and 167 a day ahead for 159 and 164.
    Beyond the weather the air was dirtier than expected by 0, 7 and 21 AQI (error breakdown).
  * the aerosol index was high for autumn (92–93rd percentile of 2021–2025 on 25–26 Oct), so some
    absorbing aerosol (dust or smoke) was there; but it was as high on 20–21 Oct with AQI 93–97,
    and in October it does not move with the daily AQI (r 0.06–0.17; NO2 0.48).
  * westerly gusts come with cleaner air in Oct–Mar (AQI 88–93 on days with westerly gusts above
    30 km/h against 125 at ≤ 20): fronts and ventilation, not dust.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import PROC, RAW, ROOT, load_target, load_weather
from aqf.satellite import LAT, LON, PASS_UTC, S5P, _keys, _km, no2_series
from aqf.silam import _session

CACHE = ROOT / "data" / "research_cache" / "aer_ai"
TIF = "PRODUCT_aerosol_index_340_380_4326.tif"
NAME = re.compile(r"S5P_(\w{4})_L2__AER_AI_(\d{8}T\d{6})_(\d{8}T\d{6})_")
BOX = (63.0, 38.5, 72.0, 44.5)  # lon/lat window read from each orbit
QA_MIN = 80
YEARS = range(2018, 2027)
BASE_FROM = 2021  # the index sits ~1.5 lower in 2018–2020 (older processing), so baselines start in 2021


def _bearing(lat, lon) -> np.ndarray:
    p1, p2, dl = np.radians(LAT), np.radians(lat), np.radians(lon - LON)
    return (np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))) + 360) % 360


def _read(key: str):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(f"/vsicurl/{S5P}/{key}") as src:
        w = from_bounds(*BOX, src.transform)
        a = src.read(1, window=w, boundless=True, fill_value=src.nodata).astype(float)
        a[a == src.nodata] = np.nan
        t = src.window_transform(w)
    rows, cols = np.indices(a.shape)
    return a, t.f + (rows + 0.5) * t.e, t.c + (cols + 0.5) * t.a


def ai_day(session, day: pd.Timestamp, product: str) -> dict | None:
    keys = _keys(session, f"COGT/{product}/L2__AER_AI/{day:%Y/%m/%d}/")
    latest = {}
    for k in keys:  # one key per granule: the latest processing wins
        m = NAME.search(k)
        if m and k.endswith(TIF) and k > latest.get(m.group(2), ""):
            latest[m.group(2)] = k
    if not latest:
        return None
    lo, hi = day + pd.Timedelta(hours=PASS_UTC[0]), day + pd.Timedelta(hours=PASS_UTC[1])
    vals, lats, lons = [], [], []
    for k in latest.values():
        m = NAME.search(k)
        t0, t1 = pd.Timestamp(m.group(2)), pd.Timestamp(m.group(3))
        if t1 < lo or t0 > hi:
            continue
        a, la, ln = _read(k)
        if not np.isfinite(a).any():
            continue
        q, _, _ = _read(k.replace(TIF, "PRODUCT_qa_value_4326.tif"))
        ok = np.isfinite(a) & (q >= QA_MIN) & (q <= 100)
        vals.append(a[ok]), lats.append(la[ok]), lons.append(ln[ok])
    out = {"date": f"{day:%Y-%m-%d}", "product": product}
    if vals:
        v, la, ln = (np.concatenate(x) for x in (vals, lats, lons))
        d, b = _km(la, ln), _bearing(la, ln)
        for name, m in (("city", d <= 30), ("region", d <= 300), ("west", (d >= 100) & (d <= 500) & (b >= 200) & (b <= 340))):
            out[name] = round(float(v[m].mean()), 3) if m.sum() >= 3 else None
            out[f"{name}_n"] = int(m.sum())
        out["region_p90"] = round(float(np.percentile(v[d <= 300], 90)), 3) if (d <= 300).sum() >= 10 else None
    return out


def fetch() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None).normalize()
    days = [d for y in YEARS for d in pd.date_range(f"{y}-09-01", f"{y}-11-30") if d < today and d >= pd.Timestamp("2018-06-28")]
    session = _session()

    def one(d):
        p = CACHE / f"{d:%Y%m%d}.json"
        if p.exists():
            return json.loads(p.read_text())
        for product in ("OFFL", "NRTI"):
            try:
                js = ai_day(session, d, product)
            except Exception as e:  # noqa: BLE001 — a day that fails is just missing
                print(f"{d:%Y-%m-%d} {product}: {type(e).__name__}", flush=True)
                js = None
            if js is not None:
                p.write_text(json.dumps(js))
                return js
        return None

    with ThreadPoolExecutor(6) as ex:
        rows = [r for r in ex.map(one, days) if r]
    return pd.DataFrame(rows).assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date").sort_index()


def daily_frame(ai: pd.DataFrame) -> pd.DataFrame:
    pm = load_target()[0]["pm"].dropna()
    a = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    g = a.resample("D")
    day = pd.DataFrame({"aqi": g.mean(), "n": g.count()})
    day = day[day["n"] >= 18].drop(columns="n")
    hr = a.index.hour
    day["night"] = a[(hr >= 21) | (hr <= 5)].resample("D").mean()  # 00–05 and 21–23 of that date
    day["afternoon"] = a[(hr >= 12) & (hr <= 17)].resample("D").mean()
    w = load_weather()
    u = -w["wind_speed_10m"] * np.sin(np.deg2rad(w["wind_direction_10m"]))
    west = w["wind_direction_10m"].between(225, 315)
    wd = pd.DataFrame({"ws": w["wind_speed_10m"].resample("D").mean(), "gust": w["wind_gusts_10m"].resample("D").max(),
                       "west_gust": w["wind_gusts_10m"].where(west, 0).resample("D").max(), "u": u.resample("D").mean(),
                       "rain": w["precipitation"].resample("D").sum(), "rh": w["relative_humidity_2m"].resample("D").min()})
    cams = pd.read_parquet(RAW / "weather" / "cams.parquet")
    no2 = no2_series()
    return (day.join(wd).join(cams["dust"].resample("D").mean().rename("cams_dust"))
            .join(no2["no2"].rename("no2")).join(ai[["city", "region", "west", "region_p90"]].add_prefix("ai_")))


def pct_rank(s: pd.Series, v: float) -> float | None:
    s = s.dropna()
    return None if pd.isna(v) or not len(s) else round(100 * float((s < v).mean()))


def episode(D: pd.DataFrame) -> dict:
    autumn = D[D.index.month.isin([9, 10, 11]) & (D.index.year >= BASE_FROM) & (D.index.year < 2026)]
    octo = D[(D.index.month == 10) & (D.index.year < 2026)]
    rows = []
    for d, r in D.loc["2026-10-18":"2026-10-27"].iterrows():
        rows.append({"date": f"{d:%Y-%m-%d}", "aqi": round(r["aqi"]), "night_over_afternoon": round(r["night"] / r["afternoon"], 2),
                     "ws": round(r["ws"], 1), "gust": round(r["gust"]), "west_gust": round(r["west_gust"]),
                     "cams_dust": round(r["cams_dust"], 1), "no2": r["no2"], "no2_pct_oct": pct_rank(octo["no2"], r["no2"]),
                     "ai_city": r["ai_city"], "ai_region": r["ai_region"], "ai_west": r["ai_west"],
                     "ai_region_pct_autumn": pct_rank(autumn["ai_region"], r["ai_region"])})
    print(pd.DataFrame(rows).to_string(index=False))
    norm = octo["night"] / octo["afternoon"]
    print(f"\nnight/afternoon AQI in past Octobers: median {norm.median():.2f}, middle half {norm.quantile(.25):.2f}–{norm.quantile(.75):.2f}")
    return {"days": rows, "night_over_afternoon_oct": [round(norm.quantile(q), 2) for q in (.25, .5, .75)]}


def by_year(D: pd.DataFrame) -> list[dict]:
    out = []
    for y in YEARS:
        g = D[(D.index.year == y) & (D.index >= f"{y}-09-20") & (D.index <= f"{y}-10-10")]
        h = D[(D.index.year == y) & (D.index >= f"{y}-10-15") & (D.index <= f"{y}-10-31")]
        out.append({"year": y, "sep20_oct10_aqi": round(g["aqi"].mean()) if g["aqi"].notna().any() else None,
                    "sep20_oct10_ai_region": round(g["ai_region"].mean(), 2) if g["ai_region"].notna().any() else None,
                    "sep20_oct10_dusty_days": int((g["ai_region"] > 1).sum()),
                    "oct15_31_aqi": round(h["aqi"].mean()) if h["aqi"].notna().any() else None,
                    "oct15_31_ai_region": round(h["ai_region"].mean(), 2) if h["ai_region"].notna().any() else None,
                    "oct15_31_no2": round(h["no2"].mean()) if h["no2"].notna().any() else None})
    print("\n" + pd.DataFrame(out).to_string(index=False))
    return out


def links(D: pd.DataFrame) -> dict:
    """Across autumn days: what moves with the daily AQI — dust (aerosol index) or burning (NO2)?"""
    A = D[D.index.month.isin([9, 10, 11]) & (D.index.year >= BASE_FROM)]
    out = {}
    for m, name in ((9, "sep"), (10, "oct"), (11, "nov"), (None, "all")):
        g = A if m is None else A[A.index.month == m]
        out[name] = {c: (round(float(g[["aqi", c]].dropna().corr("spearman").iloc[0, 1]), 2), int(g[["aqi", c]].dropna().shape[0]))
                     for c in ("ai_city", "ai_region", "ai_west", "no2", "cams_dust", "west_gust", "ws")}
    # dusty vs clear days at the same NO2 level (terciles of NO2 within each month)
    g = A.dropna(subset=["ai_region", "no2", "aqi"]).copy()
    g["no2_t"] = g.groupby(g.index.month)["no2"].transform(lambda s: pd.qcut(s, 3, labels=False, duplicates="drop"))
    g["dusty"] = g["ai_region"] > g["ai_region"].quantile(0.9)  # no autumn day of 2021–2025 reached 1
    out["dusty_vs_clear_same_no2"] = (g.groupby(["no2_t", "dusty"])["aqi"].agg(["mean", "count"]).round(1)
                                      .reset_index().to_dict("records"))
    print("\nSpearman with daily AQI (r, days):")
    print(pd.DataFrame({k: {c: v[0] for c, v in out[k].items()} for k in ("sep", "oct", "nov", "all")}).to_string())
    print("\nAQI on the 10% most aerosol-laden days vs the rest, by NO2 tercile:")
    print(pd.DataFrame(out["dusty_vs_clear_same_no2"]).to_string(index=False))
    return out


def westerly(D: pd.DataFrame) -> dict:
    """Does westerly wind bring dirtier air? Next-day change of AQI after days with a westerly gust,
    by strength, October–March; and the model's next-day errors on such days (backtest, forecast weather)."""
    W = D[D.index.month.isin([10, 11, 12, 1, 2, 3])].copy()
    W["next"] = D["aqi"].shift(-1).reindex(W.index) - W["aqi"]
    W["bin"] = pd.cut(W["west_gust"], [-1, 0.5, 20, 30, 40, 100], labels=["no west wind", "≤ 20", "20–30", "30–40", "> 40 km/h"])
    t = W.groupby("bin", observed=True).agg(days=("aqi", "size"), aqi=("aqi", "mean"), next_change=("next", "mean"),
                                            rain=("rain", "mean")).round(1)
    print("\nOctober–March: days by the strongest gust from the west (225–315°), ERA5")
    print(t.to_string())
    out = {"by_west_gust": t.reset_index().astype({"bin": str}).to_dict("records")}
    p = PROC / "bt_hourly.parquet"
    if p.exists():
        b = pd.read_parquet(p, columns=["origin", "h", "T", "pred_lp", "actual_aqi", "weather", "season"])
        b = b[(b["weather"] == "forecast") & b["h"].between(1, 24)]
        b["err"] = aqi.pm_to_aqi(np.clip(np.expm1(b["pred_lp"]), 0, None)) - b["actual_aqi"]
        e = b.groupby(b["T"].dt.normalize())["err"].mean().rename("err")
        E = W.join(e, how="inner")
        te = E.groupby("bin", observed=True)["err"].agg(["size", "mean"]).round(1)
        print("\nmodel error for the next 24 h (forecast − actual, AQI), backtest 2024–2026, by the day's westerly gust")
        print(te.to_string())
        out["model_err_by_west_gust"] = te.reset_index().astype({"bin": str}).to_dict("records")
    return out


def main() -> dict:
    ai = fetch()
    print(f"aerosol index days: {len(ai)} ({ai.index.min():%Y-%m-%d} – {ai.index.max():%Y-%m-%d}), "
          f"with a city value {ai['city'].notna().sum()}")
    D = daily_frame(ai)
    out = {"episode": episode(D), "by_year": by_year(D), "links": links(D), "westerly": westerly(D)}
    (PROC / "dust_check.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    main()
