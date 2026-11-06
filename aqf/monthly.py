"""Monthly maintenance (run by run_monthly.sh, scheduled on the 1st of every month).

1. Re-run the walk-forward backtest (monthly retraining inside every winter, current one
   included) and the report: refreshes accuracy numbers and the 80% bands.
2. Check the 80% bands against forecasts actually issued (data/forecast_log): if, with
   enough live data, the share of outcomes inside the band leaves 72–88%, the band for that
   horizon is re-estimated as an even blend of backtest and live error quantiles.
3. Save every live forecast/actual pair to data/processed/live_errors.csv and write a short
   report to logs/monthly_report_<YYYY-MM>.md, including an alarm when live errors are much
   worse than the backtest.
4. Rebuild the site (output/dashboard.html).

Model structure and the per-horizon choice of algorithms are deliberately not changed here.

    .venv/bin/python -m aqf.monthly [--skip-backtest]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import aqi
from .data import PROC, ROOT, load_target
from .models import daily_aqi

LIVE_CSV = PROC / "live_errors.csv"
INTERVALS = PROC / "intervals.json"
REPORTS = ROOT / "logs"
TARGET, LOW, HIGH = 0.80, 0.72, 0.88
MIN_HOURLY, MIN_DAILY = 300, 30  # live pairs per bucket before a band may be adjusted
ALARM = 1.20  # live MAE more than 20% above the backtest -> flag in the report


def current_season(now: pd.Timestamp) -> int:
    return now.year if now.month >= 7 else now.year - 1


def run_backtest() -> None:
    from .backtest import backtest_hourly
    from .report import summarize
    now = pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None)
    seasons = list(range(2020, current_season(now) + 1))
    t0 = time.time()
    backtest_hourly(seasons).to_parquet(PROC / "bt_hourly.parquet")
    print(f"backtest {seasons[0]}–{seasons[-1]}: {time.time() - t0:.0f}s", flush=True)
    summarize()
    from .attribution import update
    update(with_backtest=True)  # typical weather / pollution / unknown shares from the new backtest
    try:  # stagnant air vs pollution level over past winters (~3 min), for the about tab
        from .stagnation import run, summary
        summary(run())
    except Exception as e:  # noqa: BLE001 - the page keeps last month's analysis
        print("WARNING: stagnation analysis failed:", type(e).__name__, e)


def live_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every issued hourly / daily forecast value with its band and the actual outcome."""
    from .live import _issued
    pm = load_target()[0]["pm"]
    obs = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    day = daily_aqi(pm)
    h, d = [], []
    for js in _issued():
        o = pd.Timestamp(js["origin"])
        for r in js["hourly_7d"]:
            t = pd.Timestamp(r["t"])
            h.append({"origin": o, "t": t, "h": int((t - o) / pd.Timedelta(hours=1)), "pred": r["aqi"],
                      "lo": r["lo"], "hi": r["hi"], "act": obs.get(t, np.nan)})
        for i, r in enumerate(js["daily_90d"], 1):
            t = pd.Timestamp(r["t"])
            d.append({"origin": o, "t": t, "d": i, "pred": r["aqi"], "lo": r["lo"], "hi": r["hi"],
                      "act": day.get(t.normalize(), np.nan)})
    H, Dd = pd.DataFrame(h), pd.DataFrame(d)
    return (H.dropna(subset=["act"]) if len(H) else H), (Dd.dropna(subset=["act"]) if len(Dd) else Dd)


def check_bands(H: pd.DataFrame, Dd: pd.DataFrame) -> list[dict]:
    """Coverage of the 80% band per horizon bucket; re-estimate buckets that drifted."""
    bands = json.loads(INTERVALS.read_text())
    rows = []
    for key, frame, col, min_n in (("hourly", H, "h", MIN_HOURLY), ("daily", Dd, "d", MIN_DAILY)):
        for b in bands.get(key, []):
            g = frame[(frame[col] >= b["lo"]) & (frame[col] <= b["hi"])].dropna(subset=["lo", "hi"]) if len(frame) else frame
            n = int(len(g))
            cov = float(((g["act"] >= g["lo"]) & (g["act"] <= g["hi"])).mean()) if n else float("nan")
            action = "too few pairs" if n < min_n else ("ok" if LOW <= cov <= HIGH else "re-estimated")
            if action == "re-estimated":
                r = np.log1p(aqi.aqi_to_pm(g["act"].values)) - np.log1p(aqi.aqi_to_pm(g["pred"].values))
                b["q10"] = float(0.5 * b["q10"] + 0.5 * np.quantile(r, 0.1))
                b["q90"] = float(0.5 * b["q90"] + 0.5 * np.quantile(r, 0.9))
                b["live_adjusted"] = pd.Timestamp.now().strftime("%Y-%m-%d")
            rows.append({"kind": key, "bucket": f"{b['lo']}–{b['hi']}", "n": n,
                         "coverage": None if np.isnan(cov) else round(cov, 3), "action": action})
    INTERVALS.write_text(json.dumps(bands, indent=1))
    return rows


def write_report(H: pd.DataFrame, Dd: pd.DataFrame, band_rows: list[dict], did_backtest: bool) -> str:
    bt = json.loads((PROC / "backtest_summary.json").read_text())["headline"]
    now = pd.Timestamp.now(tz="Asia/Tashkent")
    lines = [f"# Monthly check {now:%Y-%m-%d}", ""]
    lines.append(f"Backtest {'recomputed' if did_backtest else 'not recomputed'}. Winter 2025/26: "
                 f"24 hours ±{bt['h24']['mae']:.1f} AQI ({bt['h24']['mape']:.0f}%), "
                 f"week ±{bt['h168']['mae']:.1f} ({bt['h168']['mape']:.0f}%).")
    lines.append("")
    alarm = []
    if len(H):
        for lab, lo, hi, ref in (("24 hours", 1, 24, bt["h24"]["mae"]), ("week", 1, 168, bt["h168"]["mae"])):
            g = H[(H["h"] >= lo) & (H["h"] <= hi)]
            if len(g):
                e = (g["pred"] - g["act"]).abs()
                lines.append(f"In real use ({g['origin'].nunique()} forecasts), {lab}: ±{e.mean():.1f} AQI "
                             f"({100 * (e / g['act']).mean():.0f}%), backtest ±{ref:.1f}.")
                if g["origin"].nunique() >= 30 and e.mean() > ALARM * ref:
                    alarm.append(f"{lab}: the real error {e.mean():.1f} is more than 20% above the backtest's {ref:.1f}")
    else:
        lines.append("No live forecast–measurement pairs yet.")
    from .live import BIAS_DAYS, score
    b = score().get("bias")
    if b:
        lines.append(f"Bias of the next-day forecast over the last {BIAS_DAYS} days ({b['days']} with measurements): "
                     f"{b['mean']:+.1f} AQI, above the measurement {b['over']} times, below {b['under']}.")
        if b["alarm"]:
            alarm.append(f"the next-day forecast is systematically {'too high' if b['mean'] > 0 else 'too low'} "
                         f"({b['mean']:+.0f} AQI on average)")
    lines += ["", "| 80% range | Horizon | Pairs | Coverage | Action |", "|---|---|---|---|---|"]
    for r in band_rows:
        cov = "—" if r["coverage"] is None else f"{100 * r['coverage']:.0f}%"
        lines.append(f"| {'hours' if r['kind'] == 'hourly' else 'days'} | {r['bucket']} | {r['n']} | {cov} | {r['action']} |")
    if alarm:
        lines += ["", "**Alarm:** " + "; ".join(alarm) + ". The model should be reviewed."]
    text = "\n".join(lines) + "\n"
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"monthly_report_{now:%Y-%m}.md").write_text(text)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-backtest", action="store_true")
    a = ap.parse_args()
    if not a.skip_backtest:
        run_backtest()
    H, Dd = live_pairs()
    if len(H) or len(Dd):
        pd.concat([H.assign(kind="hourly"), Dd.assign(kind="daily")], ignore_index=True).to_csv(LIVE_CSV, index=False)
    band_rows = check_bands(H, Dd)
    print(write_report(H, Dd, band_rows, not a.skip_backtest))
    from .dashboard import build
    print(build())


if __name__ == "__main__":
    main()
