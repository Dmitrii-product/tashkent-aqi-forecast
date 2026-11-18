"""Level shift after the 1 Dec 2026 solid-fuel ban for greenhouses? Rules fixed before the event:
docs/checks/2026-12-01-greenhouse-ban.md. Run on or after 28.12.2026, after `python -m aqf.stagnation`.

r = log(actual / forecast) of the next-day daily mean re-made from 06:00 with the actual weather by a
model fixed per winter (data/processed/stagnation.parquet). Δ = mean r over 3-21 Dec minus mean r over
10-30 Nov; the same calendar windows of winters 2020/21-2025/26 give the seasonal reference.

    PYTHONPATH=. .venv/bin/python research/dec1_check.py     # -> data/processed/dec1_check.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqf.data import PROC

YEAR = 2026
PAST = range(2020, 2026)
N_BOOT, BLOCK = 2000, 7
rng = np.random.default_rng(0)


def windows(r: pd.Series, y: int) -> tuple[pd.Series, pd.Series]:
    return r[f"{y}-11-10":f"{y}-11-30"].dropna(), r[f"{y}-12-03":f"{y}-12-21"].dropna()


def delta(r: pd.Series, y: int) -> float:
    b, a = windows(r, y)
    return float(a.mean() - b.mean())


def boot(b: pd.Series, a: pd.Series) -> tuple[float, float]:
    def resample(x: np.ndarray) -> np.ndarray:
        bl = [np.arange(i, min(i + BLOCK, len(x))) for i in range(0, len(x), BLOCK)]
        return np.concatenate([x[bl[i]] for i in rng.integers(0, len(bl), len(bl))])
    d = [resample(a.values).mean() - resample(b.values).mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    s = pd.read_parquet(PROC / "stagnation.parquet")
    s.index = pd.to_datetime(s.index).normalize()
    r = np.log(s["actual"] / s["forecast"])
    past = {y: round(delta(r, y), 4) for y in PAST}
    ref = float(np.mean(list(past.values())))
    out = {"past_delta": past, "delta_ref": round(ref, 4)}
    b, a = windows(r, YEAR)
    if len(b) < 15 or len(a) < 15:
        out["status"] = f"not enough {YEAR} days yet (before {len(b)}, after {len(a)}); run after 28.12.{YEAR}"
    else:
        d = float(a.mean() - b.mean())
        lo, hi = boot(b, a)
        out |= {"delta": round(d, 4), "ci": [round(lo, 4), round(hi, 4)], "shift": round(d - ref, 4),
                "n": [len(b), len(a)],
                "drop_found": bool(d - ref <= -0.10 and d < min(past.values()) and hi < ref),
                "rise_found": bool(d - ref >= 0.10 and d > max(past.values()) and lo > ref)}
    (PROC / "dec1_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
