"""Pick the best model variant (algorithm) separately for every horizon group.

Selection uses one winter only (2024/25); the choice is then checked on winters the
selection never saw (2025/26 and October 2026), against the best single variant.

    .venv/bin/python -m aqf.select_horizon
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import aqi
from .data import PROC
from .models import H_GROUPS

SELECT_SEASON = 2024
CHECK_SEASONS = tuple(y for y in (2025, 2026) if (PROC / f"exp_{y}_b7.parquet").exists())
TAGS = ("_b6", "_b7")
BASE = "pred_v2_mm"
CHOICE = PROC / "horizon_choice.json"


def load(y: int) -> pd.DataFrame:
    key = ["origin", "h"]
    d = None
    for t in TAGS:
        p = PROC / f"exp_{y}{t}.parquet"
        if not p.exists():
            continue
        e = pd.read_parquet(p)
        cols = key + [c for c in e.columns if c.startswith("pred_")]
        if d is None:
            d = e[key + ["actual_aqi", "persist_aqi", "clim_aqi"] + cols[2:]]
        else:
            d = d.merge(e[cols], on=key, how="inner")
    preds = [c for c in d.columns if c.startswith("pred_")]
    return d.dropna(subset=preds + ["actual_aqi"])


def group_mae(d: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = {}
    for lo, hi in H_GROUPS:
        s = d[(d["h"] >= lo) & (d["h"] <= hi)]
        rows[f"{lo}-{hi}"] = {c: np.abs(s[c] - s["actual_aqi"]).mean() for c in cols}
    return pd.DataFrame(rows).T


def routed(d: pd.DataFrame, choice: dict[str, str]) -> np.ndarray:
    out = np.full(len(d), np.nan)
    for (lo, hi) in H_GROUPS:
        m = ((d["h"] >= lo) & (d["h"] <= hi)).values
        out[m] = d.loc[m, choice[f"{lo}-{hi}"]].values
    return out


def summary(d: pd.DataFrame, cols: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = {}
    for name, p in cols.items():
        r = {}
        for lab, m in (("24h", d["h"] <= 24), ("week", d["h"] <= 168)):
            e = np.abs(p[m.values] - d.loc[m, "actual_aqi"].values)
            r[f"{lab} AQI"] = e.mean()
            r[f"{lab} %"] = 100 * (e / d.loc[m, "actual_aqi"].values).mean()
        rows[name] = r
    return pd.DataFrame(rows).T.round(1)


def main() -> None:
    sel = load(SELECT_SEASON)
    preds = [c for c in sel.columns if c.startswith("pred_")]
    g = group_mae(sel, preds)
    choice = g.idxmin(axis=1).to_dict()
    print(f"MAE by horizon group, selection winter {SELECT_SEASON}/{SELECT_SEASON + 1}:")
    print(g.round(1).to_string())
    print("\nchosen:", choice)
    for y in CHECK_SEASONS:
        d = load(y)
        best_single = d[preds].sub(d["actual_aqi"], axis=0).abs().mean().idxmin()  # hindsight
        print(f"\ncheck {y}/{y + 1} (n origins {d['origin'].nunique()}):")
        print(summary(d, {"per-horizon choice": routed(d, choice), BASE: d[BASE].values,
                          f"best single in hindsight: {best_single}": d[best_single].values}).to_string())
    CHOICE.write_text(json.dumps(choice, indent=1))


if __name__ == "__main__":
    main()
