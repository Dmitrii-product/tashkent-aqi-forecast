"""Render output/dashboard.html from the latest forecast and the backtest summary.

All interface text lives in the template's I18N dictionary (ru, en, zh, uz-Cyrillic);
this module only passes data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import safe_io
from .data import PROC, ROOT

OUT = ROOT / "output"
TEMPLATE = Path(__file__).with_name("dashboard_template.html")


def history() -> dict:
    """Daily mean AQI for the whole record, for the history tab (year vs year)."""
    from .data import load_target
    from .models import daily_aqi
    tgt, _ = load_target()
    a = daily_aqi(tgt["pm"]).dropna()
    src = tgt["src"].resample("D").mean().reindex(a.index)
    start = a.index.min()
    return {"start": f"{start:%Y-%m-%d}",
            "aqi": [None if pd.isna(v) else round(float(v)) for v in a.reindex(pd.date_range(start, a.index.max())).values],
            "iqair_from": f"{src[src > 0.5].index.min():%Y-%m-%d}"}


def build() -> Path:
    fc = json.loads((OUT / "forecast.json").read_text())
    bt = json.loads((PROC / "backtest_summary.json").read_text())
    iq = {r["b"]: r for r in bt["hourly"]["iqair"]}
    data = {
        "meta": {
            "origin": fc["origin"],
            "generated_at": fc["generated_at"],
            "latest_aqi": fc["latest_aqi"],
            "data_age_hours": fc.get("data_age_hours", 0),
        },
        "obs": [[r["t"], r["aqi"]] for r in fc["recent"]],
        "h": [[r["t"], r["aqi"], r["lo"], r["hi"]] for r in fc["hourly_7d"]],
        "d": [[r["t"], r["aqi"], r["lo"], r["hi"], r["clim_aqi"], r["kind"], r.get("p_red"), r.get("p_purple")]
              for r in fc["daily_90d"]],
        "months": fc.get("months", {}),
        "why": fc.get("explain", []),  # SHAP by group of inputs (aqf.explain), week tab
        "hist": history(),
        "bt": {
            "headline": bt["headline"],
            "hourly": bt["hourly"],
            "daily": bt["daily"],
            "seasons": bt["seasons"],
            "n_forecasts": bt["n_forecasts"],
            "d24": {"model": iq["13–24"]["model"], "persist": iq["13–24"]["persist"]},
            "d7": {"model": iq["121–168"]["model"], "clim": iq["121–168"]["clim"]},
        },
    }
    try:
        from .live import score
        data["live"] = score()
    except Exception as e:  # noqa: BLE001 - the page must still build
        data["live"] = {"n_forecasts": 0, "days": [], "error": type(e).__name__}
    try:  # model versions and the first / last forecast each issued: what changed and since when
        from .live import versions
        from .models import FIRST_VERSION, GBM_VERSION, MODEL_VERSION
        data["model"] = {"current": MODEL_VERSION, "forecast": fc.get("model_version", FIRST_VERSION),
                         "backtest": bt.get("model_version", FIRST_VERSION), "gbm": GBM_VERSION, "versions": versions()}
    except Exception as e:  # noqa: BLE001 - the page must still build
        data["model"] = {"error": type(e).__name__}
    attr = PROC / "attribution.json"  # error breakdown (aqf.attribution), forecast-vs-fact tab
    data["attr"] = json.loads(attr.read_text()) if attr.exists() else {}
    stag = PROC / "stagnation.json"  # stagnant air vs pollution level over past winters (aqf.stagnation)
    data["stag"] = json.loads(stag.read_text()) if stag.exists() else {}
    try:  # SILAM (FMI) vs our forecast vs fact, other-forecasts tab
        from .silam import score as silam_score
        data["silam"] = silam_score()
    except Exception as e:  # noqa: BLE001 - the page must still build
        data["silam"] = {"error": type(e).__name__}
    try:  # emission-cut scenario (aqf.scenario): its own tab, never mixed into the forecast
        from .data import load_target
        from .scenario import build as scenario
        data["scen"] = scenario(fc, load_target()[0]["pm"])
    except Exception as e:  # noqa: BLE001 - the page must still build
        data["scen"] = {"error": type(e).__name__}
    try:  # Sentinel-5P NO2 and FIRMS fires next to the fact (aqf.satellite), satellites tab
        from .satellite import score as satellite_score
        data["sat"] = satellite_score()
    except Exception as e:  # noqa: BLE001 - the page must still build
        data["sat"] = {"error": type(e).__name__}
    try:  # how the week tiles' red-day mark did on past winters (aqf.report), week tab
        from .report import red_week
        data["red_week"] = red_week()
    except Exception as e:  # noqa: BLE001 - the page must still build
        data["red_week"] = {"error": type(e).__name__}
    nn_bt = PROC / "nn_test.json"  # the network vs LightGBM on past winters (aqf.nn), next to its shadow scores
    data["nn_bt"] = [r for r in json.loads(nn_bt.read_text())["rows"] if r["hours"] in ("1-24", "1-168")] if nn_bt.exists() else []
    html = TEMPLATE.read_text().replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c"))
    out = OUT / "dashboard.html"
    safe_io.write_text(out, html)
    return out


if __name__ == "__main__":
    print(build())
