"""A small transformer instead of / next to the GRU of aqf/nn.py (owner, 2026-11-07).

Rules fixed before the first run: docs/checks/2026-11-07-transformer.md.

Only the encoder of the past 72 h changes: a linear projection of the 16 past channels to 64, a learned
position for each of the 72 hours, 2 pre-norm encoder layers (4 heads, feed-forward 128, dropout 0.1),
and the vector of the last hour (the origin) as the summary, like the GRU's last state. Inputs, the
168-hour decoder, loss, weights, early stopping, 3 seeds and the backtest protocol (aqf.nn.run:
monthly retraining, origins every 3 h, archived forecast weather) are the network's; learning rate
1e-3 instead of 2e-3 (transformers without warm-up are unstable at high rates).

  A  transformer instead of the GRU: LightGBM 1-6 h, 50/50 LightGBM + transformer from 7 h
  B  both networks: LightGBM 1-6 h, 0.5 LightGBM + 0.25 GRU + 0.25 transformer from 7 h
Chosen on winter 2024/25 (MAE over 1-168 h, must beat the current blend there); passes if lower than
the current blend on 2025/26 and October 2026 with the 95% interval over days entirely below zero.

    PYTHONPATH=. .venv/bin/python research/transformer_test.py           # 3 seeds + scoring
    PYTHONPATH=. .venv/bin/python research/transformer_test.py --score   # scoring only

Writes data/processed/tf_test_s{0,1,2}.parquet and transformer_test.json (never the network's own files).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import aqf.nn as gru
from aqf import aqi
from aqf.backtest import season_bounds
from aqf.blend import BLEND_FROM_H
from aqf.data import PROC, load_prev_runs_ensemble, load_weather, splice_forecast_weather
from aqf.features import weather_features
from research.no2_input_test import boot

REF = PROC / "nn_blend_v2026-10-28.parquet"
BANDS = {"1-24": (1, 24), "25-72": (25, 72), "73-168": (73, 168), "1-168": (1, 168)}
SEEDS = (0, 1, 2)


class TNet(nn.Module):
    def __init__(self, c_past: int, c_fut: int, c_stat: int, hid: int = 64):
        super().__init__()
        self.inp = nn.Linear(c_past, hid)
        self.pos = nn.Parameter(torch.zeros(1, gru.PAST, hid))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(hid, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hid)
        self.stat = nn.Sequential(nn.Linear(c_stat, hid), nn.GELU())
        self.hemb = nn.Embedding(gru.HOR, 16)
        self.dec = nn.Sequential(nn.Linear(2 * hid + c_fut + 16, 128), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, past, fut, stat):
        z = self.norm(self.enc(self.inp(past) + self.pos))[:, -1]
        e = torch.cat([z, self.stat(stat)], dim=-1)[:, None, :].expand(-1, gru.HOR, -1)
        he = self.hemb(torch.arange(gru.HOR, device=past.device))[None].expand(past.shape[0], -1, -1)
        return self.dec(torch.cat([e, fut, he], dim=-1)).squeeze(-1)


def run(seed: int) -> pd.DataFrame:
    """aqf.nn.run with TNet and lr 1e-3; writes its own file."""
    _AdamW, _Net = torch.optim.AdamW, gru.Net
    gru.Net = TNet  # aqf.nn.train builds `Net(...)` from the module namespace
    torch.optim.AdamW = lambda params, lr, weight_decay: _AdamW(params, lr=1e-3, weight_decay=weight_decay)  # noqa: E731
    try:
        wx = load_weather()
        wx_act = weather_features(wx)
        wx_fc = splice_forecast_weather(wx, load_prev_runs_ensemble())
        preds = []
        for y in gru.SEASONS:
            start, end = season_bounds(y)
            cutoffs = pd.date_range(start, end, freq="MS", inclusive="left")
            for i, cut in enumerate(cutoffs):
                nxt = cutoffs[i + 1] if i + 1 < len(cutoffs) else end
                t0 = time.time()
                data = gru.Data(cut, wx_act, wx_fc)
                pos = data.test_pos(nxt)
                if not len(pos):
                    continue
                model = gru.train(data, seed=seed)
                preds.append(gru.predict(model, data, pos))
                print(f"seed {seed} {cut:%Y-%m}: {len(pos)} forecasts, {time.time() - t0:.0f}s", flush=True)
    finally:
        torch.optim.AdamW, gru.Net = _AdamW, _Net
    df = pd.concat(preds, ignore_index=True)
    df.to_parquet(PROC / f"tf_test_s{seed}.parquet")
    return df



def score() -> dict:
    runs = [pd.read_parquet(PROC / f"tf_test_s{k}.parquet") for k in SEEDS]
    tf = runs[0][["origin", "h"]].assign(tf_lp=np.mean([r["nn_lp"].values for r in runs], axis=0))
    df = pd.read_parquet(REF).merge(tf, on=["origin", "h"], how="inner")
    lp = lambda a: np.log1p(aqi.aqi_to_pm(a))  # noqa: E731
    to_aqi = lambda x: aqi.pm_to_aqi(np.clip(np.expm1(x), 0, None))  # noqa: E731
    g, n, t = lp(df["gbm"].values), lp(df["nn"].values), df["tf_lp"].values
    late = (df["h"] >= BLEND_FROM_H).values
    df["current"] = df["blend7"]
    df["tf"] = to_aqi(t)
    df["A"] = to_aqi(np.where(late, 0.5 * g + 0.5 * t, g))
    df["B"] = to_aqi(np.where(late, 0.5 * g + 0.25 * n + 0.25 * t, g))
    mae = lambda d, k: round(float(np.abs(d[k] - d["actual"]).mean()), 2)  # noqa: E731
    keys = ("gbm", "nn", "tf", "current", "A", "B")
    out = {"pairs": int(len(df)), "mae": {}, "vs_current": {}}
    for y in (2024, 2025, 2026):
        d = df[df["season"] == y]
        out["mae"][y] = {k: {b: mae(d[d["h"].between(*r)], k) for b, r in BANDS.items()} for k in keys}
        out["vs_current"][y] = {v: {b: [round(x, 2) for x in boot(d[d["h"].between(*r)], "current", v)] for b, r in BANDS.items()}
                                for v in ("A", "B")}
    m24 = out["mae"][2024]
    chosen = min(("A", "B"), key=lambda v: m24[v]["1-168"])
    out["chosen"] = chosen if m24[chosen]["1-168"] < m24["current"]["1-168"] else None
    out["passes"] = bool(out["chosen"] and all(out["vs_current"][y][out["chosen"]]["1-168"][2] < 0 for y in (2025, 2026)))
    (PROC / "transformer_test.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if not a.score:
        for k in SEEDS:
            run(k)
    print(json.dumps(score(), indent=1))


if __name__ == "__main__":
    main()
