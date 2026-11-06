"""Should the network's share in the served blend differ by horizon? (owner, 2026-11-05)

Since model version 2026-11-04 the hourly forecast is LightGBM for hours 1–6 and the 50/50 blend
(in log PM2.5) of LightGBM and the network from hour 7 (aqf/blend.py). The 50/50 was not tuned.
Here the share is chosen from the backtest both models already have: data/processed/
nn_blend_v2026-10-28.parquet (aqf.backtest protocol, monthly retraining, archived forecast weather;
the pairs the version was adopted on). No model is trained or written.

  one       — one network share w for all hours from 7 (hours 1–6 stay LightGBM)
  by_group  — a share for each of the 14 horizon groups of aqf.models.H_GROUPS, hours 1–6 included

Rules fixed before the run:
  * shares on a grid 0, 0.05, …, 1, chosen by MAE (AQI) on winter 2024/25 only;
  * the candidate is `by_group` (`one` is reported alongside); it passes only if its MAE over
    1–168 h is lower than the current blend's on both 2025/26 and October 2026 (to 26.10, the end of
    the file) with the 95% bootstrap interval over days entirely below zero; a pass is a new model
    version (shares in aqf/blend.py), shown on the site.

    PYTHONPATH=. .venv/bin/python research/blend_weight_test.py

Writes data/processed/blend_weight_test.json.

Result (2026-11-05), MAE change vs the current blend over 1–168 h (95% interval over days):
                 2024/25 (choice)       2025/26                Oct 2026 (to 26.10)
  one (w 0.35)   -0.06 (-0.17..+0.04)   +0.08 (-0.04..+0.21)   +0.58 (+0.25..+0.88)
  by_group       -0.07 (-0.17..+0.02)   +0.05 (-0.07..+0.17)   +0.54 (+0.28..+0.77)   <- candidate
  by_group shares chosen on 2024/25: 0 at 1–3 h, 0.1 at 4–6, 0.2–0.45 from 7 h.
Rejected: a lower network share wins a little on 2024/25 only. The share each period would pick for
itself grows with time: ~0.35–0.45 on 2024/25, 0.45–0.65 on 2025/26, 0.55–1.0 in October 2026 — the
network has been gaining on LightGBM, so 50/50 stays; to be looked at again after the winter.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.blend import BLEND_FROM_H
from aqf.data import PROC
from aqf.models import H_GROUPS
from research.no2_input_test import boot

GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
BANDS = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}


def lp(a: np.ndarray) -> np.ndarray:
    return np.log1p(aqi.aqi_to_pm(a))


def to_aqi(x: np.ndarray) -> np.ndarray:
    return aqi.pm_to_aqi(np.clip(np.expm1(x), 0, None))


def main() -> dict:
    df = pd.read_parquet(PROC / "nn_blend_v2026-10-28.parquet")
    g, n = lp(df["gbm"].values), lp(df["nn"].values)
    cur = to_aqi(np.where(df["h"] >= BLEND_FROM_H, 0.5 * (g + n), g))
    assert np.nanmax(np.abs(cur - df["blend7"].values)) < 0.5, "AQI -> log PM round trip does not rebuild the blend"
    df["current"] = df["blend7"]
    grp = np.full(len(df), -1)
    for i, (lo, hi) in enumerate(H_GROUPS):
        grp[df["h"].between(lo, hi).values] = i
    sel = (df["season"] == 2024).values
    err = lambda w: np.abs(to_aqi((1 - w) * g + w * n) - df["actual"].values)  # noqa: E731
    E = {w: err(w) for w in GRID}
    late = (df["h"] >= BLEND_FROM_H).values
    w_one = min(GRID, key=lambda w: np.where(late, E[w], E[0.0])[sel].mean())
    w_grp = {f"{lo}-{hi}": float(min(GRID, key=lambda w: E[w][sel & (grp == i)].mean())) for i, (lo, hi) in enumerate(H_GROUPS)}
    W = np.array([w_grp[f"{lo}-{hi}"] for lo, hi in H_GROUPS])[grp]
    df["one"] = to_aqi(np.where(late, (1 - w_one) * g + w_one * n, g))
    df["by_group"] = to_aqi((1 - W) * g + W * n)
    mae = lambda d, k: round(float(np.abs(d[k] - d["actual"]).mean()), 2)  # noqa: E731
    out = {"w_one": float(w_one), "w_by_group": w_grp, "mae": {}, "diff_vs_current": {}}
    for y in (2024, 2025, 2026):
        d = df[df["season"] == y]
        out["mae"][y] = {k: {b: mae(d[d["h"].between(*r)], k) for b, r in BANDS.items()} for k in ("gbm", "current", "one", "by_group")}
        out["diff_vs_current"][y] = {k: {b: [round(x, 2) for x in boot(d[d["h"].between(*r)], "current", k)] for b, r in BANDS.items()}
                                     for k in ("one", "by_group")}
    # the share each unseen period would have chosen for itself (not used for anything)
    out["w_by_group_if_chosen_on"] = {y: {f"{lo}-{hi}": float(min(GRID, key=lambda w: E[w][(df["season"] == y).values & (grp == i)].mean()))
                                          for i, (lo, hi) in enumerate(H_GROUPS)} for y in (2025, 2026)}
    d25, d26 = (out["diff_vs_current"][y]["by_group"]["1-168"] for y in (2025, 2026))
    out["passes"] = bool(d25[2] < 0 and d26[2] < 0)
    print(json.dumps(out, indent=1))
    (PROC / "blend_weight_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
