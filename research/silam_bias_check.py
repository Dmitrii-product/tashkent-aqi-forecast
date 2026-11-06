"""Does SILAM under-forecast Tashkent because it does not see the pollution that is already there?

SILAM's global forecast does not assimilate measurements: each run starts from the previous
day's forecast (FMI, ICAP 2023). If that were the whole story, SILAM would be wrong at the start
but its emissions would pull it towards the real level within a day or two; if its emissions
(CAMS-GLOB-ANT on a 20 km grid) are too low, it stays low however long it runs. Checks on the
32 saved runs (26.09–27.10.2026) against the measured PM2.5:

1. Error by lead time: is SILAM already low in its first hours, and does that change by day 4?
2. Daily cycle: does it reproduce the evening and night peaks of local sources under inversions?
3. Week by week: does the gap grow as heating starts (World Bank: heating share 3% of PM2.5 in
   September, 15% in October, 38% in November)?
4. Its own 3×3 neighbourhood: is the city a hot spot in SILAM at all?
5. What the current measurement adds: SILAM's next-day forecast as is, shifted by the gap of
   the last 24 h, scaled by the ratio of the last 7 days (both known at issue time), against
   "tomorrow = the last 24 h" and against our forecast.
6. CAMS (assimilates satellite aerosol, not ground stations) over the same hours, for contrast.

Result (2026-10-27). SILAM does start without the real pollution, but that is not why it stays
low: its own sources are too weak for Tashkent.
- Measured / SILAM PM2.5 is 2.3 in the first 6 hours and 2.0–2.4 out to 5 days; the error stays
  ±40–48 AQI at every lead. A wrong start that its emissions could fix would shrink with lead.
- No night peak: measured night (20–01 h) / afternoon (13–16 h) is 1.47, SILAM 0.88 (CAMS 2.86).
  The evening and night build-up of the city's own smoke under inversions is missing; the
  Tashkent cell is no hot spot either (3×3 neighbourhood / cell = 1.01).
- The gap grows as heating starts: ratio 2.1–2.6 through mid-October, 3.3 in the week from
  26.10 (gap 51 µg/m³ against 15–23).
- The current measurement supplies most of what SILAM lacks (next day's mean AQI, 16 days
  05–26.10): SILAM as is ±47.9 (all of it bias), shifted by the last 24 h gap ±24.4, scaled by
  the last 7 days' ratio ±21.3, "tomorrow = last 24 h" ±25.5. On the 8 days with our forecast:
  ±52.1 / ±20.6 / ±18.3 / ±21.4, ours ±20.9 (running +20 high) — too few days to rank.
- CAMS, which assimilates satellite aerosol, is low too: measured / CAMS 1.3–5.0 by winter
  month, 2.26 on average (Oct–Feb 2022–2026) — global emission inventories miss Tashkent's
  winter sources, it is not SILAM alone.

    PYTHONPATH=. .venv/bin/python research/silam_bias_check.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqf import aqi, silam
from aqf.data import ROOT, load_target
from aqf.live import _by_day, issue_day

H = pd.Timedelta(hours=1)
CAMS = ROOT / "data" / "raw" / "weather" / "cams.parquet"


def a(pm):
    return aqi.pm_to_aqi(pm)


def main() -> None:
    pm = load_target()[0]["pm"]
    cams = pd.read_parquet(CAMS)["pm2_5"]
    runs = silam.load()

    # every SILAM value with its lead from the run start (00 UTC = 05:00 Tashkent)
    rows = []
    for r, df in runs.items():
        t0 = pd.Timestamp(r) + pd.Timedelta(hours=5)
        rows.append(pd.DataFrame({"run": r, "t": df.index, "lead": (df.index - t0) / H,
                                  "silam": df["pm25"].values, "s3": df["pm25_3x3"].values}))
    S = pd.concat(rows, ignore_index=True)
    S["act"], S["cams"] = pm.reindex(S["t"]).values, cams.reindex(S["t"]).values
    S = S.dropna(subset=["act"])
    S = S[S["t"] >= "2026-10-01"]  # from here on every hour is covered by all five lead days

    print("1. By lead time (hours from the run start), 1–26 Oct, PM2.5 µg/m³")
    S["lb"] = pd.cut(S["lead"], [0, 6, 24, 48, 72, 96, 120], labels=["1-6", "7-24", "25-48", "49-72", "73-96", "97-120"])
    g = S.groupby("lb", observed=True)
    tab = pd.DataFrame({"n": g.size(), "measured": g["act"].mean(), "silam": g["silam"].mean(),
                        "ratio": g["act"].mean() / g["silam"].mean(),
                        "silam_mae_aqi": g.apply(lambda d: np.abs(a(d["silam"]) - a(d["act"])).mean(), include_groups=False),
                        "r": g.apply(lambda d: np.corrcoef(d["silam"], d["act"])[0, 1], include_groups=False)})
    print(tab.round(2).to_string(), "\n")

    # the newest run for each hour (lead 1–24): what SILAM said for that hour on the day
    D = S[(S["lead"] >= 1) & (S["lead"] <= 24)].copy()
    print("2. Daily cycle, 1–26 Oct (lead 1–24 h), mean PM2.5 by hour of day")
    D["hod"] = D["t"].dt.hour
    cyc = D.groupby("hod")[["act", "silam", "cams"]].mean()
    print(cyc.T.round(0).to_string())
    for c in ("act", "silam", "cams"):
        night, day = cyc.loc[[20, 21, 22, 23, 0, 1], c].mean(), cyc.loc[[13, 14, 15, 16], c].mean()
        print(f"   {c:6s} night 20-01 / afternoon 13-16 = {night / day:.2f}")
    print()

    print("3. Week by week (lead 1–24 h), mean PM2.5")
    D["week"] = D["t"].dt.to_period("W-SUN").dt.start_time
    w = D.groupby("week")[["act", "silam", "cams"]].mean()
    w["gap"], w["ratio"], w["cams_ratio"] = w["act"] - w["silam"], w["act"] / w["silam"], w["act"] / w["cams"]
    print(w.round(1).to_string(), "\n")

    print("4. SILAM 3×3 neighbourhood mean / Tashkent cell:", round(float((D["s3"] / D["silam"]).mean()), 2), "\n")

    print("5. Next day's mean AQI, forecast on the morning before (the SILAM run of that morning)")
    day_act = pd.Series(a(pm.values), index=pm.index).groupby(pm.index.normalize()).agg(["mean", "count"])
    day_act = day_act["mean"].where(day_act["count"] == 24)
    ours = {}
    for _, js in _by_day():
        d0 = issue_day(js)
        for row in js["daily_90d"]:
            if pd.Timestamp(row["t"]) == d0 + pd.Timedelta(days=1):
                ours[d0 + pd.Timedelta(days=1)] = row["aqi"]
    rows = []
    for r, df in runs.items():
        d0 = pd.Timestamp(r)
        day = d0 + pd.Timedelta(days=1)
        if np.isnan(day_act.get(day, np.nan)):
            continue
        issue = d0 + pd.Timedelta(hours=6)  # what was measured by the time of the forecast
        last24 = pm[issue - 23 * H: issue]
        # SILAM's own values for those 24 h, from the run of the day before
        prev = [x for x in runs if x < r]
        if len(last24) < 20 or not prev:
            continue
        s_last24 = runs[max(prev)]["pm25"].reindex(last24.index)
        # ratio over the previous 7 days, each hour by the run of its own day
        hist = D[(D["t"] > issue - pd.Timedelta(days=7)) & (D["t"] <= issue)]
        fc = df["pm25"][(df.index >= day) & (df.index < day + pd.Timedelta(days=1))]
        rows.append({"day": day, "act": day_act[day], "silam": a(fc).mean(),
                     "silam_shift": a(np.clip(fc + (last24.mean() - s_last24.mean()), 0, None)).mean(),
                     "silam_ratio7": a(fc * hist["act"].mean() / hist["silam"].mean()).mean() if len(hist) >= 72 else np.nan,
                     "last24": a(last24).mean(), "ours": ours.get(day, np.nan)})
    R = pd.DataFrame(rows).set_index("day")
    cols = ["silam", "silam_shift", "silam_ratio7", "last24"]
    base = R.dropna(subset=cols)
    print(f"   all days with every SILAM variant ({len(base)}, {base.index.min():%d.%m}–{base.index.max():%d.%m}):")
    for c in cols:
        e = base[c] - base["act"]
        print(f"   {c:13s} MAE {e.abs().mean():5.1f}  bias {e.mean():+6.1f}  r {np.corrcoef(base[c], base['act'])[0, 1]:.2f}")
    both = R.dropna(subset=cols + ["ours"])
    print(f"   days with our forecast too ({len(both)}, {both.index.min():%d.%m}–{both.index.max():%d.%m}):")
    for c in cols + ["ours"]:
        e = both[c] - both["act"]
        print(f"   {c:13s} MAE {e.abs().mean():5.1f}  bias {e.mean():+6.1f}")
    print()

    print("6. CAMS vs measured, by month (all hours), measured / CAMS:")
    m = pd.DataFrame({"act": pm, "cams": cams}).dropna()
    m = m[m.index >= "2022-10-01"]
    mm = m.groupby(m.index.to_period("M")).mean()
    mm["ratio"] = mm["act"] / mm["cams"]
    print(mm[mm.index.month.isin([10, 11, 12, 1, 2])].round(2).T.to_string())
    print(json.dumps({"cams_ratio_oct_feb_mean": round(float(mm[mm.index.month.isin([10, 11, 12, 1, 2])]["ratio"].mean()), 2)}))


if __name__ == "__main__":
    main()
