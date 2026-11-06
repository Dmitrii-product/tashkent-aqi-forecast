"""Did Tashkent airport's own weather forecast (TAF UZTT) see the calm of 28.10–3.11 2026 better
than the weather models? (owner, 2026-11-04: "we were badly wrong one day ahead too")

METAR is what the airport measured; its forecast is the TAF: 24 h, issued every 6 h (05, 11, 17,
23 UTC), prevailing wind in knots, visibility, phenomena (FU smoke, HZ haze, BR mist), TEMPO/PROB
for temporary changes. No open archive for Uzbekistan (Iowa Mesonet keeps US TAFs only, Ogimet has
none); aviationweather.gov serves the last 30 days. So this is one month, not a model input test.

At our 06:00 issue (01 UTC) the latest TAF is the one of 23 UTC, valid 05:00 today – 05:00
tomorrow: it covers our first ~23 hours, not the next calendar day.

  1. fetch: every TAF of the last 30 days (raw text to data/raw/weather/taf_UZTT/, written once);
  2. our error in those first 23 hours and on the next calendar day, per issued forecast;
  3. over the same window of each issue day: the TAF's prevailing wind and the 4-model mean wind
     as issued (`<stem>_wx.parquet`) against the wind measured at the airport (METAR), and smoke /
     haze in the TAF against the AQI.

    PYTHONPATH=. .venv/bin/python research/taf_check.py [fetch]

Result (2026-11-04, 109 TAFs of 6.10–4.11, 13 issued forecasts 22.10–3.11):
  * we were off badly in the first 23 hours too: 28.10 -32, 31.10 -23, 2.11 +40 (next calendar day
    28.10 -29, 29.10 -29, 31.10 -24, 1.11 -23, 2.11 +37).
  * the TAF's prevailing wind is a template: only 8 or 10 kt (15–18.5 km/h) in a month, 10 km/h above
    the measured 5.7–9.0; day to day it does not follow the airport's wind (r 0.05; all 27 days of
    23 UTC issues 0.01), the 4-model mean does (r 0.72). On 28.10 (-32) the TAF promised 10 kt,
    showers and thunderstorms, visibility 7000 m, no smoke.
  * smoke or haze (FU/HZ) in the TAF goes with the AQI (r 0.46; daily, all 27 days 0.30) — what the
    forecaster sees at issue time, which the model already has from the measurements.
  No use as a weather forecast for us; not collected.
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

from aqf import aqi
from aqf.data import PROC, RAW, load_target, load_weather
from aqf.live import issued_stems

TAF_DIR = RAW / "weather" / "taf_UZTT"
LOG = PROC.parent / "forecast_log"
METAR = RAW / "weather" / "metar_UZTT_full.csv"
AWC = "https://aviationweather.gov/api/data/taf"
KT = 1.852


def fetch() -> None:
    TAF_DIR.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now("UTC").tz_localize(None).floor("h")
    for t in pd.date_range(now - pd.Timedelta(days=29, hours=18), now, freq="6h"):
        r = requests.get(AWC, params={"ids": "UZTT", "format": "raw", "date": f"{t:%Y-%m-%dT%H:%M:%SZ}"}, timeout=60)
        txt = r.text.strip()
        m = re.search(r"UZTT (\d{2})(\d{2})(\d{2})Z", txt)
        if r.ok and m:
            issued = pd.Timestamp(year=t.year, month=t.month, day=int(m[1]), hour=int(m[2]), minute=int(m[3]))
            if issued > t + pd.Timedelta(days=1):  # day of the previous month
                issued -= pd.offsets.MonthBegin(1)
            p = TAF_DIR / f"{issued:%Y%m%d%H%M}.txt"
            if not p.exists():
                p.write_text(txt + "\n")
        time.sleep(1)


def _when(ddhh: str, ref: pd.Timestamp) -> pd.Timestamp:
    """'2816' -> 28th 16 UTC of the month of `ref` (or the next one); hour 24 = next day 00."""
    d, h = int(ddhh[:2]), int(ddhh[2:4])
    t = pd.Timestamp(year=ref.year, month=ref.month, day=1)
    if d < ref.day - 15:
        t += pd.offsets.MonthBegin(1)
    return t + pd.Timedelta(days=d - 1, hours=h)


def parse(txt: str, issued: pd.Timestamp) -> pd.DataFrame:
    """Hourly (UTC) prevailing wind (km/h), visibility (m), and smoke/haze anywhere (also TEMPO)."""
    tok = txt.replace("\n", " ").split()
    i = next(k for k, s in enumerate(tok) if re.fullmatch(r"\d{4}/\d{4}", s))
    start, end = _when(tok[i][:4], issued), _when(tok[i][5:], issued)
    idx = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    out = pd.DataFrame(index=idx, data={"ws": np.nan, "vis": np.nan, "smoke": 0.0, "vis_min": np.nan})
    group, g_from, g_to = "BASE", start, end
    for s in tok[i + 1:]:
        if s in ("TEMPO", "BECMG") or s.startswith("PROB"):
            group = s if s != "TEMPO" or not group.startswith("PROB") else group
            continue
        if re.fullmatch(r"FM\d{6}", s):
            group, g_from, g_to = "BASE", _when(s[2:6], issued), end
            continue
        if re.fullmatch(r"\d{4}/\d{4}", s):
            g_from, g_to = _when(s[:4], issued), _when(s[5:], issued)
            continue
        sel = (out.index >= (g_to if group == "BECMG" else g_from)) & (out.index < (end if group == "BECMG" else g_to))
        temp = group == "TEMPO" or group.startswith("PROB")
        w = re.fullmatch(r"(VRB|\d{3})(\d{2,3})(G\d{2,3})?KT", s)
        if w and not temp:
            out.loc[sel, "ws"] = int(w[2]) * KT
        v = re.fullmatch(r"(\d{4})|CAVOK", s)
        if v:
            vis = 10000 if s == "CAVOK" else int(s)
            if not temp:
                out.loc[sel, "vis"] = vis
            out.loc[sel, "vis_min"] = np.fmin(out.loc[sel, "vis_min"], vis)
        if re.search(r"(FU|HZ)$", s):
            out.loc[sel, "smoke"] = 1.0
    out["vis_min"] = np.fmin(out["vis_min"], out["vis"])
    return out


def tafs() -> dict[pd.Timestamp, pd.DataFrame]:
    out = {}
    for p in sorted(TAF_DIR.glob("*.txt")):
        issued = pd.Timestamp(p.stem)
        f = parse(p.read_text(), issued)
        f.index = f.index + pd.Timedelta(hours=5)  # to Asia/Tashkent, as everywhere in aqf
        out[issued + pd.Timedelta(hours=5)] = f
    return out


def main() -> None:
    pm = load_target()[0]["pm"]
    obs = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    m = pd.read_csv(METAR, na_values="M")
    met = (m.set_index(pd.to_datetime(m["valid"]))["sknt"] * KT).resample("h").mean()
    era = load_weather()["wind_speed_10m"]
    T = tafs()
    rows = []
    for s in issued_stems():
        js = json.loads((LOG / f"{s}.json").read_text())
        o = pd.Timestamp(js["origin"])
        cand = [k for k in T if k <= o]
        if not cand or not (LOG / f"{s}_wx.parquet").exists():
            continue
        taf = T[max(cand)]
        h = pd.DataFrame(js.get("hourly_7d_gbm", js["hourly_7d"]))
        h = h.set_index(pd.to_datetime(h["t"]))["aqi"]
        win = h.index[(h.index > o) & (h.index <= o + pd.Timedelta(hours=23))]
        nxt = h.index[h.index.normalize() == o.normalize() + pd.Timedelta(days=1)]
        wx = pd.read_parquet(LOG / f"{s}_wx.parquet")["wind_speed_10m"]
        tw = taf.reindex(win)
        rows.append({"issued": s, "taf": f"{max(cand):%d %H}", "err_23h": (h[win] - obs.reindex(win)).mean(),
                     "err_next_day": (h[nxt] - obs.reindex(nxt)).mean(), "aqi_23h": obs.reindex(win).mean(),
                     "airport": met.reindex(win).mean(), "taf_ws": tw["ws"].mean(), "models": wx.reindex(win).mean(),
                     "era5": era.reindex(win).mean(), "taf_vis": tw["vis_min"].min(), "taf_smoke_h": tw["smoke"].sum()})
    df = pd.DataFrame(rows).set_index("issued")
    pd.set_option("display.width", 200)
    print("our error (hours 1-23 / next calendar day), AQI; wind km/h over hours 1-23: airport measured,"
          " TAF prevailing, 4-model mean as issued, ERA5; TAF lowest visibility (m) and smoke/haze hours")
    print(df.round(1).to_string())
    print("\nTAF minus airport", round((df["taf_ws"] - df["airport"]).mean(), 1), " models minus ERA5",
          round((df["models"] - df["era5"]).mean(), 1))
    for k in ("taf_ws", "models"):
        print(f"day-to-day: corr({k}, airport) {df[k].corr(df['airport']):.2f}   corr({k}, AQI) {df[k].corr(df['aqi_23h']):.2f}")
    print(f"corr(TAF smoke/haze hours, AQI) {df['taf_smoke_h'].corr(df['aqi_23h']):.2f}")
    # the whole month of TAFs: prevailing wind against the airport, the 23 UTC issues
    allt = pd.concat([f.assign(issued=k) for k, f in T.items() if k.hour == 4])
    allt = allt[~allt.index.duplicated(keep="last")]
    d = pd.DataFrame({"taf": allt["ws"], "smoke": allt["smoke"], "airport": met.reindex(allt.index),
                      "aqi": obs.reindex(allt.index)}).dropna()
    D = d.resample("D").mean().dropna()
    print(f"\nall 23 UTC TAFs ({len(D)} days): daily wind corr with airport {D['taf'].corr(D['airport']):.2f};"
          f" smoke/haze share corr with AQI {D['smoke'].corr(D['aqi']):.2f}; distinct prevailing speeds (kt):"
          f" {sorted(set((allt['ws'] / KT).round().dropna().astype(int)))}")


if __name__ == "__main__":
    if "fetch" in sys.argv[1:]:
        fetch()
    main()
