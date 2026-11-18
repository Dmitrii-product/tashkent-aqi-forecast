"""Does Chronos-2, used as is (zero-shot), add anything to the served forecast? (owner, 2026-11-07)

Rules fixed before the first run: docs/checks/2026-11-07-chronos2-zero-shot.md.

Chronos-2 (amazon/chronos-2 on Hugging Face, ~120 M parameters) is a pretrained time-series model.
It is not trained on our data here. For every (origin, hour) pair of data/processed/
nn_blend_v2026-10-28.parquet (origins every 3 h; winters 2024/25 and 2025/26, 1-26 Oct 2026) it reads
  * log(1 + PM2.5), hourly, IQAir scale: load_target(profile_cutoff = the origin's month start), as
    the backtest; the last 2048 h up to and including the origin hour, gaps left as gaps;
  * variant "wx" (the candidate): the 12 weather inputs WX of aqf/nn.py as covariates — actual
    weather (reanalysis) up to the origin, then the archived forecast of the matching lead day
    (1-7), as the network and LightGBM get it in the backtest;
  * variant "plain" (reported only): the series alone;
and forecasts 168 h; the point forecast is the median, in log PM2.5.

Scoring: new = (1 - w) * current + w * Chronos-2 (log PM2.5), current = the served blend (LightGBM
for 1-6 h, 50/50 with the network from 7 h); one share w for 1-168 h on the grid 0..1 by 0.05,
chosen on winter 2024/25 only. Passes if MAE over 1-168 h is lower than current on both 2025/26 and
October 2026 with the 95% bootstrap interval over days entirely below zero.

Needs chronos-forecasting (research only; not in requirements): uv pip install --python .venv/bin/python chronos-forecasting

    PYTHONPATH=. .venv/bin/python research/chronos_zero_shot.py            # forecasts + scoring
    PYTHONPATH=. .venv/bin/python research/chronos_zero_shot.py --score    # scoring only
    PYTHONPATH=. .venv/bin/python research/chronos_zero_shot.py --limit 40 # timing on a few origins

Writes data/processed/chronos_zs.parquet and chronos_zero_shot.json.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.backtest import season_bounds
from aqf.data import PROC, load_prev_runs_ensemble, load_target, load_weather, splice_forecast_weather
from aqf.features import weather_features
from aqf.nn import WX
from research.no2_input_test import boot

CTX, HOR = 2048, 168
GRID = np.round(np.arange(0, 1.0001, 0.05), 2)
BANDS = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
OUT = PROC / "chronos_zs.parquet"
REF = PROC / "nn_blend_v2026-10-28.parquet"


def lp(a: np.ndarray) -> np.ndarray:
    return np.log1p(aqi.aqi_to_pm(a))


def to_aqi(x: np.ndarray) -> np.ndarray:
    return aqi.pm_to_aqi(np.clip(np.expm1(x), 0, None))


def forecasts(limit: int | None = None) -> pd.DataFrame:
    import torch
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps" if torch.backends.mps.is_available() else "cpu",
                                            dtype=torch.float32)
    q50 = [0.5]
    origins = pd.DatetimeIndex(sorted(pd.read_parquet(REF, columns=["origin"])["origin"].unique()))
    if limit:
        origins = origins[:: max(1, len(origins) // limit)][:limit]
    wx = load_weather()
    act = weather_features(wx)[WX]
    fc = {d: f[WX] for d, f in splice_forecast_weather(wx, load_prev_runs_ensemble()).items()}
    H = np.arange(1, HOR + 1)
    lead = (H - 1) // 24 + 1
    out = []
    months = origins.to_period("M")
    for m in months.unique():
        t0 = time.time()
        cut = m.to_timestamp()
        os_ = origins[months == m]
        tgt, _ = load_target(profile_cutoff=cut)
        idx = pd.date_range(tgt.index.min(), os_.max() + pd.Timedelta(hours=HOR), freq="h")
        y = np.log1p(tgt["pm"].reindex(idx)).values
        A = act.reindex(idx)
        A = A.interpolate(limit=6).fillna(A[idx < cut].mean()).values  # covariates must be complete
        F = np.stack([fc[d].reindex(idx).values for d in range(1, 8)])  # (lead day, hour, WX)
        wx_in, plain_in = [], []
        for o in os_:
            p = idx.get_loc(o)
            s = max(0, p - CTX + 1)
            ctx = y[s:p + 1]
            T = p + H
            fut = F[lead - 1, T]  # (HOR, len(WX)): each hour from the forecast of its lead day
            fut = np.where(np.isnan(fut), A[T], fut)  # rare gaps in the archive: fall back to the filled series
            wx_in.append({"target": ctx,
                          "past_covariates": {c: A[s:p + 1, i] for i, c in enumerate(WX)},
                          "future_covariates": {c: fut[:, i] for i, c in enumerate(WX)}})
            plain_in.append(ctx)
        qw, _ = pipe.predict_quantiles(wx_in, prediction_length=HOR, quantile_levels=q50, batch_size=13 * 24)
        qp, _ = pipe.predict_quantiles(plain_in, prediction_length=HOR, quantile_levels=q50, batch_size=256)
        for o, a, b in zip(os_, qw, qp):
            out.append(pd.DataFrame({"origin": o, "h": H, "chr_wx": a[0, :, 0].cpu().numpy(), "chr_plain": b[0, :, 0].cpu().numpy()}))
        print(f"{m}: {len(os_)} origins, {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(out, ignore_index=True)
    if not limit:
        df.to_parquet(OUT)
    return df


def score(ch: pd.DataFrame) -> dict:
    df = pd.read_parquet(REF).merge(ch, on=["origin", "h"], how="inner")
    cur = lp(df["blend7"].values)
    df["current"] = df["blend7"]
    df["chronos_wx"], df["chronos_plain"] = to_aqi(df["chr_wx"].values), to_aqi(df["chr_plain"].values)
    sel = (df["season"] == 2024).values
    err = lambda w: np.abs(to_aqi((1 - w) * cur + w * df["chr_wx"].values) - df["actual"].values)  # noqa: E731
    w = float(min(GRID, key=lambda v: err(v)[sel].mean()))
    df["mix"] = to_aqi((1 - w) * cur + w * df["chr_wx"].values)
    mae = lambda d, k: round(float(np.abs(d[k] - d["actual"]).mean()), 2)  # noqa: E731
    keys = ("gbm", "nn", "current", "chronos_wx", "chronos_plain", "mix")
    out = {"w": w, "pairs": int(len(df)), "mae": {}, "mix_vs_current": {}}
    for y in (2024, 2025, 2026):
        d = df[df["season"] == y]
        out["mae"][y] = {k: {b: mae(d[d["h"].between(*r)], k) for b, r in BANDS.items()} for k in keys}
        out["mix_vs_current"][y] = {b: [round(x, 2) for x in boot(d[d["h"].between(*r)], "current", "mix")] for b, r in BANDS.items()}
    # the share each unseen period would have chosen for itself (not used for anything)
    out["w_if_chosen_on"] = {y: float(min(GRID, key=lambda v: err(v)[(df["season"] == y).values].mean())) for y in (2025, 2026)}
    d25, d26 = (out["mix_vs_current"][y]["1-168"] for y in (2025, 2026))
    out["passes"] = bool(w > 0 and d25[2] < 0 and d26[2] < 0)
    out["fine_tune_next"] = bool(w > 0)
    (PROC / "chronos_zero_shot.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.limit:
        t0 = time.time()
        df = forecasts(a.limit)
        print(f"{df['origin'].nunique()} origins in {time.time() - t0:.0f}s")
        return
    ch = pd.read_parquet(OUT) if a.score else forecasts()
    print(json.dumps(score(ch), indent=1))


if __name__ == "__main__":
    main()
