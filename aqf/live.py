"""Forecast vs reality in real use: every issued forecast (data/forecast_log) against what
then happened. Nothing is back-filled: only forecasts actually issued before the day count.

    .venv/bin/python -m aqf.live
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import aqi
from .data import ROOT, load_target
from .models import FIRST_VERSION, MODEL_VERSIONS, daily_aqi

LOG = ROOT / "data" / "forecast_log"
LEADS = (1, 3, 7)  # days ahead for the daily comparison
TABLE_DAYS = 30
# drift watch: the 1-day-ahead daily forecast is flagged as biased when, over the last 14 days
# (at least 7 scored), its mean error is 15+ AQI and 75%+ of the days miss the same way. On the
# 2024–2026 backtest this fired on 12% of days, about four episodes a winter, each a real one.
BIAS_DAYS, BIAS_MIN_N, BIAS_AQI, BIAS_SHARE = 14, 7, 15.0, 0.75


def issue_day(js: dict) -> pd.Timestamp:
    """The day a forecast was published (Tashkent time). "Days ahead" count from it, not from
    the origin: after an overnight data gap the morning forecast can start from yesterday's
    last hour, and must still count as today's forecast."""
    return pd.Timestamp(js["generated_at"]).tz_localize(None).normalize()


def _by_day() -> list[tuple[str, dict]]:
    """(file stem, forecast) of the first forecast issued on each calendar day (the morning run)."""
    by_day: dict = {}
    for f in sorted(LOG.glob("??????????.json")):  # <origin>.json, not the <origin>_nn.json shadows
        js = json.loads(f.read_text())
        by_day.setdefault(issue_day(js), (f.stem, js))
    return [by_day[k] for k in sorted(by_day)]


def _issued() -> list[dict]:
    """The first forecast issued on each calendar day (the morning routine run)."""
    return [js for _, js in _by_day()]


def issued_stems() -> list[str]:
    return [st for st, _ in _by_day()]


def version_of(js: dict) -> str:
    """The model version (aqf.models.MODEL_VERSIONS) that issued a forecast; stamped since the
    second version, so an issued forecast without the stamp is from the first."""
    return js.get("model_version", FIRST_VERSION)


def versions() -> list[dict]:
    """Every model version with the first and the last morning forecast it issued (None: none yet)."""
    seen: dict = {}
    for _, js in _by_day():
        r = seen.setdefault(version_of(js), {"first": js["generated_at"], "n": 0})
        r["last"], r["n"] = js["generated_at"], r["n"] + 1
    return [mv | seen.get(mv["id"], {"first": None, "last": None, "n": 0}) for mv in MODEL_VERSIONS]


def _hourly_mae(h: pd.DataFrame) -> dict:
    out = {}
    for lo, hi, name in ((1, 24, "day"), (1, 168, "week")):
        s = h[(h["h"] >= lo) & (h["h"] <= hi)]
        if len(s) >= 24:
            e = (s["pred"] - s["act"]).abs()
            out[name] = {"mae": float(e.mean()), "mape": float((e / s["act"]).mean() * 100), "n": int(len(s))}
    return out


def score() -> dict:
    by_day = _by_day()
    issued = [js for _, js in by_day]
    if not issued:
        return {"n_forecasts": 0, "days": []}
    pm = load_target()[0]["pm"]
    obs = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    day_act = daily_aqi(pm)
    hrows, drows = [], []
    for js in issued:
        o, v = pd.Timestamp(js["origin"]), version_of(js)
        for r in js["hourly_7d"]:
            t = pd.Timestamp(r["t"])
            hrows.append({"h": int((t - o) / pd.Timedelta(hours=1)), "pred": r["aqi"], "act": obs.get(t, np.nan), "v": v})
        today = issue_day(js)
        for r in js["daily_90d"]:
            t = pd.Timestamp(r["t"]).normalize()
            lead = (t - today).days
            if lead in LEADS:
                drows.append({"date": t, "lead": lead, "pred": r["aqi"], "issued": today, "v": v})
    h = pd.DataFrame(hrows).dropna()
    out = {"n_forecasts": len(issued), "first": str(pd.Timestamp(issued[0]["origin"]))} | _hourly_mae(h)
    # the same for each model version once a newer one has issued forecasts: the site shows the
    # newest version's errors next to the overall ones
    if h["v"].nunique() > 1:
        out["by_version"] = {v: _hourly_mae(g) | {"n_forecasts": sum(version_of(js) == v for js in issued)}
                             for v, g in h.groupby("v")}
    # per target day: actual daily mean and what was forecast 1 / 3 / 7 days before
    d = pd.DataFrame(drows)
    days = []
    if len(d):
        last = day_act.dropna().index.max()
        for t in sorted(d["date"].unique()):
            t = pd.Timestamp(t)
            if t > last:
                continue
            act = day_act.get(t, np.nan)
            rec = {"date": f"{t:%Y-%m-%d}", "act": None if np.isnan(act) else round(float(act))}
            for L in LEADS:
                m = d[(d["date"] == t) & (d["lead"] == L)]
                rec[f"f{L}"] = int(m["pred"].iloc[0]) if len(m) else None
                rec[f"v{L}"] = m["v"].iloc[0] if len(m) else None
            days.append(rec)
        days = days[-TABLE_DAYS:]
        for L in LEADS:
            e = [abs(r[f"f{L}"] - r["act"]) for r in days if r[f"f{L}"] is not None and r["act"] is not None]
            if e:
                out[f"daily_lead{L}"] = {"mae": float(np.mean(e)), "n": len(e)}
    out["days"] = days
    out["bias"] = drift(days)
    out["shadow"] = shadow(by_day, obs, day_act)
    return out


def shadow(by_day: list[tuple[str, dict]], obs: pd.Series, day_act: pd.Series) -> dict | None:
    """The network and the LightGBM + network blend, forecast in the shadow next to the issued
    forecasts (aqf.nn shadow), against LightGBM: MAE on exactly the hours and days all three have."""
    hrows, drows, n, n_bf = [], [], 0, 0
    last = day_act.dropna().index.max()
    for stem, js in by_day:
        f = LOG / f"{stem}_nn.json"
        if not f.exists():
            continue
        sh = json.loads(f.read_text())
        n, n_bf = n + 1, n_bf + bool(sh.get("backfilled"))
        o = pd.Timestamp(js["origin"])
        nn = {r["t"]: r for r in sh["hourly_7d"]}
        # since the blend is served (model version 2026-11-04) LightGBM's own forecast is kept next to it
        for r in js.get("hourly_7d_gbm", js["hourly_7d"]):
            if r["t"] in nn:
                t = pd.Timestamp(r["t"])
                hrows.append({"h": int((t - o) / pd.Timedelta(hours=1)), "gbm": r["aqi"], "nn": nn[r["t"]]["nn"],
                              "blend": nn[r["t"]]["blend"], "act": obs.get(t, np.nan)})
        gbm_day = ({pd.Timestamp(k): v for k, v in js["daily_gbm"].items()} if "daily_gbm" in js
                   else {pd.Timestamp(r["t"]).normalize(): r["aqi"] for r in js["daily_90d"]})
        for r in sh["daily"]:
            t = pd.Timestamp(r["t"])
            lead = (t - issue_day(js)).days
            if lead in (1, 3) and t <= last and t in gbm_day:
                drows.append({"lead": lead, "gbm": gbm_day[t], "nn": r["nn"], "blend": r["blend"], "act": day_act.get(t, np.nan)})
    if not n:
        return None
    out = {"n": n, "backfilled": n_bf}
    mae = lambda s: {k: round(float((s[k] - s["act"]).abs().mean()), 1) for k in ("gbm", "nn", "blend")} | {"n": int(len(s))}  # noqa: E731
    h = pd.DataFrame(hrows).dropna()
    for lo, hi, name in ((1, 24, "day"), (1, 168, "week")):
        s = h[(h["h"] >= lo) & (h["h"] <= hi)] if len(h) else h
        if len(s) >= 24:
            out[name] = mae(s)
    d = pd.DataFrame(drows).dropna()
    for L in (1, 3):
        s = d[d["lead"] == L] if len(d) else d
        if len(s):
            out[f"lead{L}"] = mae(s)
    return out


def drift(days: list[dict]) -> dict | None:
    """Signed error of the 1-day-ahead forecasts over the last BIAS_DAYS days, with the alarm flag."""
    scored = [r for r in days if r["f1"] is not None and r["act"] is not None]
    if not scored:
        return None
    last = pd.Timestamp(scored[-1]["date"])
    e = np.array([r["f1"] - r["act"] for r in scored if pd.Timestamp(r["date"]) > last - pd.Timedelta(days=BIAS_DAYS)])
    mean, same = float(e.mean()), float(max((e > 0).mean(), (e < 0).mean()))
    return {"days": int(len(e)), "mean": round(mean, 1), "over": int((e > 0).sum()), "under": int((e < 0).sum()),
            "to": f"{last:%Y-%m-%d}",
            "alarm": bool(len(e) >= BIAS_MIN_N and abs(mean) >= BIAS_AQI and same >= BIAS_SHARE)}


if __name__ == "__main__":
    print(json.dumps(score(), ensure_ascii=False, indent=1))
