"""Why a forecast missed: each day's error split into weather, pollution and unknown.

What is split is the error of the forecast the site showed: the daily mean of day D as issued
on the morning of day D−L (L = 1 or 3 days ahead). It is re-run, with the model that issued it,
and compared with two variants:

  weather   = the re-run with the weather forecast it had − the same run with the weather
              that actually happened (Open-Meteo archive)
  pollution = that run with actual weather − a reference run from D 00:00 with actual weather.
              The reference starts at midnight of the day, when the model already knows how
              polluted the air really was by then: the difference is how much dirtier or
              cleaner the air got than expected even with the right weather (sources,
              transport, build-up)
  unknown   = the reference run's own error: what even it, with actual weather, missed
  re-run    = issued forecast − its re-run (a different model version or rebuilt weather;
              about zero for forecasts that saved their inputs)
  network   = since model version 2026-11-04: the blend the site showed − LightGBM's own forecast
              (saved with it as daily_gbm); the parts above split LightGBM's. Re-running the
              network for each variant would take minutes per day, so its shift is kept apart

The parts add up to the issued forecast minus the actual daily mean (positive = forecast too
high). Seven days ahead is not split: that far the forecast is climatology, the weather does
not enter it. Past winters (the backtest, same scheme from 06:00 origins, re-run = 0) give
the typical sizes; days in real use are added as they complete. Writes
data/processed/attribution.json for the site.

    .venv/bin/python -m aqf.attribution          # backtest summary + days in real use
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from . import aqi
from .data import PROC, ROOT, load_prev_runs_ensemble, load_target, load_weather, splice_forecast_weather
from .features import weather_features
from .models import daily_aqi

OUT = PROC / "attribution.json"
LOG = ROOT / "data" / "forecast_log"
MODELS, MODELS_PREV = PROC / "models.pkl", PROC / "models_prev.pkl"
CAMS = ROOT / "data" / "raw" / "weather" / "cams.parquet"
LEADS = (1, 3)
PARTS = ("weather", "pollution", "unknown")
RECOMPUTE_DAYS = 7  # recent days are redone each run: the archive weather may still be revised


def _split(issued: float, fw: float, aw: float, mid: float, act: float) -> dict:
    return {"error": issued - act, "weather": fw - aw, "pollution": aw - mid, "unknown": mid - act,
            "rerun": issued - fw}


# ---------------------------------------------------------------- past winters (backtest)

def backtest() -> dict:
    b = pd.read_parquet(PROC / "bt_hourly.parquet",
                        columns=["origin", "h", "T", "pred_aqi", "actual_aqi", "weather"])
    b["date"] = b["T"].dt.normalize()
    hourly_act = b.drop_duplicates("T").set_index("T")["actual_aqi"]
    g = hourly_act.groupby(hourly_act.index.normalize()).agg(["mean", "size"])
    act = g["mean"].where(g["size"] >= 18)

    def day_mean(sel: pd.DataFrame) -> pd.Series:
        g = sel.groupby("date")["pred_aqi"].agg(["mean", "size"])
        return g["mean"].where(g["size"] >= 18)

    aw_rows = b[b["weather"] == "actual"]
    m = aw_rows[(aw_rows["origin"] == aw_rows["date"]) & (aw_rows["h"] <= 23)]
    g = m.groupby("date")["pred_aqi"].agg(["sum", "size"])
    mid = ((hourly_act.reindex(g.index).values + g["sum"]) / (g["size"] + 1)).where(g["size"] >= 18)
    out = {}
    for lead in LEADS:
        s = b[(b["origin"].dt.hour == 6) & ((b["date"] - b["origin"].dt.normalize()).dt.days == lead)]
        df = pd.DataFrame({"fw": day_mean(s[s["weather"] == "forecast"]), "aw": day_mean(s[s["weather"] == "actual"]),
                           "mid": mid, "act": act}).dropna()
        parts = pd.DataFrame([_split(r.fw, r.fw, r.aw, r.mid, r.act) for r in df.itertuples()], index=df.index)
        out[str(lead)] = {"days": len(df), "from": f"{df.index.min():%Y-%m-%d}", "to": f"{df.index.max():%Y-%m-%d}",
                          **{k: round(float(parts[k].abs().mean()), 1) for k in ("error",) + PARTS}}
    return out


# ---------------------------------------------------------------- days in real use

class _Models:
    """The current and the previous model (kept by aqf.forecast on retrain), loaded lazily."""

    def __init__(self, current: dict | None):
        self._m = {MODELS: current}

    def _get(self, path):
        if self._m.get(path) is None and path.exists():
            self._m[path] = pickle.loads(path.read_bytes())
        return self._m.get(path)

    def pick(self, issued_at: pd.Timestamp, day: pd.Timestamp) -> tuple[dict | None, bool]:
        """The model that issued the forecast (newest trained by `issued_at`), else the newest
        one that has not seen `day`; and whether it is the issuing one."""
        ms = [m for m in (self._get(MODELS), self._get(MODELS_PREV)) if m is not None]
        ms.sort(key=lambda m: pd.Timestamp(m["trained_at"]), reverse=True)
        for m in ms:
            if pd.Timestamp(m["trained_at"]) <= issued_at:
                return m, True
        for m in ms:
            if pd.Timestamp(m["trained_at"]) < day:
                return m, False
        return None, False


def live(models: dict | None = None) -> list[dict]:
    from .live import _issued, issue_day
    old = {f"{r['issued']}|{r['lead']}": r for r in json.loads(OUT.read_text()).get("live", [])} if OUT.exists() else {}
    pm = load_target()[0]["pm"]
    day_act = daily_aqi(pm)
    arch = load_weather()
    wf_act = weather_features(arch)
    covered = arch.index.max()
    cams_hist = pd.read_parquet(CAMS) if CAMS.exists() else None
    today = pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None).normalize()
    store = _Models(models)
    recon = None  # issue-time weather rebuilt from archived forecasts, for forecasts saved without it
    rows = []
    for js in _issued():
        o = pd.Timestamp(js["origin"])
        issued_at = pd.Timestamp(js["generated_at"]).tz_localize(None)
        issued_by_day = {pd.Timestamp(r["t"]).normalize(): r["aqi"] for r in js["daily_90d"]}
        gbm_by_day = {pd.Timestamp(k): v for k, v in js.get("daily_gbm", {}).items()}
        stem = LOG / f"{o:%Y%m%d%H}"
        for lead in LEADS:
            D = issue_day(js) + pd.Timedelta(days=lead)
            key = f"{o:%Y-%m-%d %H:%M}|{lead}"
            act = day_act.get(D, np.nan)
            if D + pd.Timedelta(hours=23) > covered or np.isnan(act) or D not in issued_by_day:
                continue  # the day is not over yet, or its weather is not archived yet
            if key in old and "rerun" in old[key] and D < today - pd.Timedelta(days=RECOMPUTE_DAYS):
                rows.append(old[key])
                continue
            m, same_model = store.pick(issued_at, D)
            if m is None:  # no model that has not seen the day: keep what was computed before
                if key in old:
                    rows.append(old[key])
                continue
            router = m["hourly"]
            wx_saved, cams_saved = stem.with_name(stem.name + "_wx.parquet"), stem.with_name(stem.name + "_cams.parquet")
            if wx_saved.exists():
                wf_fc = weather_features(pd.read_parquet(wx_saved))
            else:
                if recon is None:
                    recon = splice_forecast_weather(arch, load_prev_runs_ensemble())
                wf_fc = recon

            def run(origin, wf, horizons, cams):
                if cams is not None and hasattr(router, "set_cams"):
                    router.set_cams(cams)
                return router.predict_frame(pd.DatetimeIndex([origin]), pm, wf, horizons)["pred_aqi"]

            hours = pd.date_range(D, D + pd.Timedelta(hours=23), freq="h")
            hz = ((hours - o) / pd.Timedelta(hours=1)).astype(int).values
            cams_iss = pd.read_parquet(cams_saved) if cams_saved.exists() else cams_hist  # what the forecast had
            fw = run(o, wf_fc, hz, cams_iss).mean()
            aw = run(o, wf_act, hz, cams_iss).mean()
            # reference from midnight (CAMS as it turned out). The 00:00 value is the hour that
            # ends at midnight, the first of the day as tashkentairdata.uz counts days; like the
            # model's own state it may come from 1-2 h earlier, and without it the day is skipped
            at0 = pm.ffill(limit=2).get(D, np.nan)
            if np.isnan(at0):
                if key in old:
                    rows.append(old[key])
                continue
            midp = run(D, wf_act, np.arange(1, 24), cams_hist)
            mid = (float(aqi.pm_to_aqi(at0)) + midp.sum()) / (len(midp) + 1)
            gbm = gbm_by_day.get(D, issued_by_day[D])
            parts = _split(gbm, fw, aw, mid, act) | {"error": issued_by_day[D] - act, "network": issued_by_day[D] - gbm}
            rows.append({"date": f"{D:%Y-%m-%d}", "lead": lead, "issued": f"{o:%Y-%m-%d %H:%M}",
                         "forecast": int(issued_by_day[D]), "actual": round(float(act)),
                         **{k: round(float(v), 1) for k, v in parts.items()},
                         "weather_saved": wx_saved.exists(), "same_model": same_model})
    return sorted(rows, key=lambda r: (r["date"], r["lead"]))


def update(models: dict | None = None, with_backtest: bool = False) -> dict:
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    if with_backtest or "backtest" not in data:
        data["backtest"] = backtest()
    data["live"] = live(models)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return data


if __name__ == "__main__":
    d = update(with_backtest=True)
    print(json.dumps(d["backtest"], ensure_ascii=False))
    for r in d["live"]:
        print(r)
