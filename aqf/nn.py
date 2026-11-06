"""A neural network next to the hourly LightGBM models: head-to-head on past winters.

Sequence model: a GRU reads the last 72 h of pollution and actual weather; for each of the
next 168 hours a shared decoder takes that summary, the weather forecast for that hour (as
archived at the matching lead), the calendar, the seasonal norm (leave-own-season-out in
training, as the LightGBM models), and the ventilation and rain accumulated since the origin.
It predicts the change of log PM2.5 from the level at the origin, as the LightGBM models do,
with an L1 loss and the same cold-season / IQAir weights.

Protocol = the backtest's (aqf.backtest): retrained at every month start on all data before
it, forecasts from every 3rd hour of the winters 2024/25, 2025/26 and October 2026, archived
weather forecasts as input. Scored on exactly the backtest's (origin, hour) pairs against the
production LightGBM forecasts, and as a 50/50 blend with them (in log PM2.5).

    .venv/bin/python -m aqf.nn            # 3 seeds -> data/processed/nn_test_s*.parquet, nn_test.json
    .venv/bin/python -m aqf.nn shadow     # next to the latest issued forecast, if it has no network part yet
    .venv/bin/python -m aqf.nn serve --origin T --wx F --out J   # for aqf.forecast (model version 2026-11-04)

Since model version 2026-11-04 the site's forecast is the blend from BLEND_FROM_H on (aqf.blend):
aqf.forecast runs `serve` in its own process, so it never imports PyTorch and falls back to
LightGBM alone if the network fails. Shadow mode (run_daily.sh, after the forecast) fills in the
network and the blend for a forecast issued without them; both are scored in real use on the
the other-forecasts tab. Needs PyTorch (requirements-nn.txt).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import aqi
from .backtest import season_bounds
from .blend import BLEND_FROM_H, nn_record
from .data import PROC, load_prev_runs_ensemble, load_target, load_weather, splice_forecast_weather
from .features import Climatology, season_of, weather_features
from .models import _weights

WX = ["t2m", "rh", "ws", "u", "v", "precip", "cloud", "psfc", "sw", "dT925", "dT850", "ws850"]
PAST, HOR, STEP_TRAIN, STEP_TEST = 72, 168, 2, 3
SEASONS = (2024, 2025, 2026)
BUCKETS = [(1, 3), (4, 6), (7, 12), (13, 24), (25, 48), (49, 72), (73, 120), (121, 168)]
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Data:
    """Hourly matrices for one cutoff; batches are gathered from them by index."""

    def __init__(self, cut: pd.Timestamp, wx_act: pd.DataFrame, wx_fc: dict[int, pd.DataFrame],
                 wx_live: pd.DataFrame | None = None):
        tgt, _ = load_target(profile_cutoff=cut)
        pm, src = tgt["pm"], tgt["src"]
        start = pm.dropna().index.min()
        self.idx = idx = pd.date_range(start, pm.index.max() + pd.Timedelta(days=8), freq="h")
        self.cut = cut
        self.cut_pos = int(idx.searchsorted(cut))
        lp = np.log1p(pm.reindex(idx))
        self.lp = lp.values
        self.lag0 = lp.ffill(limit=2).values  # the model's state at the origin, as LightGBM
        train = idx < cut
        # past channels: level (filled), observed flag, 24 h / 168 h means, actual weather
        lpf = lp.ffill(limit=6)
        mu_lp, sd_lp = np.nanmean(lpf[train]), np.nanstd(lpf[train])
        norm = lambda a: np.nan_to_num((a - mu_lp) / sd_lp)  # noqa: E731
        A = wx_act.reindex(idx)[WX]
        live = wx_live.reindex(idx)[WX] if wx_live is not None else None
        if live is not None:  # a live forecast: past hours as the issued forecast saw them, archive before
            A = live.combine_first(A)
        self.wmu, self.wsd = A[train].mean().values, A[train].std().values + 1e-6
        P = [norm(lpf.values), lp.notna().values.astype(float),
             norm(lp.rolling(24, min_periods=12).mean().values), norm(lp.rolling(168, min_periods=84).mean().values)]
        P += list(np.nan_to_num(((A.values - self.wmu) / self.wsd)).T)
        self.P = np.stack(P, axis=1).astype(np.float32)
        # future weather by lead day (1..7), raw and normalised
        self.Fraw = np.stack([wx_fc[d].reindex(idx)[WX].values for d in range(1, 8)])
        if live is not None:  # after the origin, every lead takes the weather forecast it was issued with
            self.Fraw[:, idx > cut, :] = live.values[idx > cut]
        self.F = np.nan_to_num((self.Fraw - self.wmu) / self.wsd).astype(np.float32)
        self.Fraw = np.nan_to_num(self.Fraw)
        h, doy, dow = idx.hour.values, idx.dayofyear.values, idx.dayofweek.values
        self.CAL = np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                             np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                             np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], axis=1).astype(np.float32)
        # seasonal norm: from data before the cutoff; leave-own-season-out for training rows
        clim = Climatology().fit(lp[train])
        self.seasons = np.unique(season_of(idx[train]))
        self.CL_loo = np.stack([norm(clim.at(idx, np.full(len(idx), s))) for s in self.seasons]).astype(np.float32)
        self.CL = norm(clim.at(idx)).astype(np.float32)
        self.season_pos = np.searchsorted(self.seasons, season_of(idx))
        self.w = _weights(idx, src.reindex(idx).values).astype(np.float32)
        self.mu_lp, self.sd_lp = mu_lp, sd_lp
        ok = ~np.isnan(self.lag0)
        pos = np.arange(len(idx))
        self.train_pos = pos[ok & train & (pos >= PAST) & (idx.hour % STEP_TRAIN == 0)]

    def test_pos(self, end: pd.Timestamp) -> np.ndarray:
        pos = np.arange(len(self.idx))
        last = min(end, pd.Timestamp(self.idx[~np.isnan(self.lp)].max()))
        m = (self.idx >= self.cut) & (self.idx < last) & (self.idx.hour % STEP_TEST == 0) & ~np.isnan(self.lag0)
        return pos[m]

    def batch(self, pos: np.ndarray, loo: bool):
        H = np.arange(1, HOR + 1)
        past = self.P[pos[:, None] + np.arange(-PAST + 1, 1)[None, :]]
        T = pos[:, None] + H[None, :]
        lead = ((H - 1) // 24)[None, :]
        fw = self.F[lead, T]
        ws, pr = self.Fraw[lead, T, WX.index("ws")], self.Fraw[lead, T, WX.index("precip")]
        vent = (np.cumsum(ws, axis=1) / H - self.wmu[WX.index("ws")]) / self.wsd[WX.index("ws")]
        rain = np.log1p(np.cumsum(np.clip(pr, 0, None), axis=1))
        cl = self.CL_loo[np.clip(self.season_pos[pos], 0, len(self.seasons) - 1)[:, None], T] if loo else self.CL[T]
        fut = np.concatenate([fw, self.CAL[T], cl[..., None], vent[..., None], rain[..., None]], axis=2).astype(np.float32)
        cl_o = self.CL_loo[np.clip(self.season_pos[pos], 0, len(self.seasons) - 1), pos] if loo else self.CL[pos]
        l0 = np.nan_to_num((self.lag0[pos] - self.mu_lp) / self.sd_lp)
        stat = np.stack([l0, cl_o, l0 - cl_o, self.CAL[pos, 0], self.CAL[pos, 1]], axis=1).astype(np.float32)
        y = (self.lp[T] - self.lag0[pos][:, None])
        m = (~np.isnan(y) & (T < self.cut_pos)) if loo else ~np.isnan(y)  # training targets end at the cutoff
        w = np.where(m, self.w[np.clip(T, 0, len(self.w) - 1)], 0.0).astype(np.float32)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)  # noqa: E731
        return t(past), t(fut), t(stat), t(np.nan_to_num(y).astype(np.float32)), t(w)


class Net(nn.Module):
    def __init__(self, c_past: int, c_fut: int, c_stat: int, hid: int = 64):
        super().__init__()
        self.gru = nn.GRU(c_past, hid, batch_first=True)
        self.stat = nn.Sequential(nn.Linear(c_stat, hid), nn.GELU())
        self.hemb = nn.Embedding(HOR, 16)
        self.dec = nn.Sequential(nn.Linear(2 * hid + c_fut + 16, 128), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, past, fut, stat):
        _, hN = self.gru(past)
        e = torch.cat([hN[-1], self.stat(stat)], dim=-1)[:, None, :].expand(-1, HOR, -1)
        he = self.hemb(torch.arange(HOR, device=past.device))[None].expand(past.shape[0], -1, -1)
        return self.dec(torch.cat([e, fut, he], dim=-1)).squeeze(-1)


def train(data: Data, seed: int = 0, epochs: int = 30, patience: int = 4, bs: int = 512) -> Net:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pos = data.train_pos
    n_val = int(0.08 * len(pos))
    tr, va = pos[:-n_val], pos[-n_val:]  # validation = the last weeks before the cutoff
    p, f, s, *_ = data.batch(tr[:2], loo=True)
    model = Net(p.shape[-1], f.shape[-1], s.shape[-1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_of = lambda z, y, w: (w * (z - y).abs()).sum() / w.sum().clamp(min=1)  # noqa: E731
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        for i in range(0, len(perm), bs):
            p, f, s, y, w = data.batch(perm[i:i + bs], loo=True)
            opt.zero_grad()
            loss_of(model(p, f, s), y, w).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tot = num = 0.0
            for i in range(0, len(va), 2048):
                p, f, s, y, w = data.batch(va[i:i + 2048], loo=True)
                tot += float((w * (model(p, f, s) - y).abs()).sum()); num += float(w.sum())
        v = tot / max(num, 1)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def predict(model: Net, data: Data, pos: np.ndarray) -> pd.DataFrame:
    out = []
    with torch.no_grad():
        for i in range(0, len(pos), 2048):
            b = pos[i:i + 2048]
            p, f, s, *_ = data.batch(b, loo=False)
            z = model(p, f, s).cpu().numpy()
            lp = z + data.lag0[b][:, None]
            out.append(pd.DataFrame({"origin": np.repeat(data.idx[b], HOR), "h": np.tile(np.arange(1, HOR + 1), len(b)),
                                     "nn_lp": lp.ravel()}))
    return pd.concat(out, ignore_index=True)


def run(seasons=SEASONS, seed: int = 0) -> pd.DataFrame:
    wx = load_weather()
    wx_act = weather_features(wx)
    wx_fc = splice_forecast_weather(wx, load_prev_runs_ensemble())
    preds = []
    for y in seasons:
        start, end = season_bounds(y)
        cutoffs = pd.date_range(start, end, freq="MS", inclusive="left")
        for i, cut in enumerate(cutoffs):
            nxt = cutoffs[i + 1] if i + 1 < len(cutoffs) else end
            t0 = time.time()
            data = Data(cut, wx_act, wx_fc)
            pos = data.test_pos(nxt)
            if not len(pos):
                continue
            model = train(data, seed=seed)
            preds.append(predict(model, data, pos))
            print(f"{cut:%Y-%m}: {len(data.train_pos)} training origins, {len(pos)} forecasts, {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(preds, ignore_index=True)
    df.to_parquet(PROC / f"nn_test_s{seed}.parquet")
    return df


def compare(nn_df: pd.DataFrame, per_seed: list[pd.DataFrame] | None = None) -> dict:
    """MAE (AQI) of LightGBM, the network and their 50/50 blend on the same pairs; with several
    seeds, `nn_df` is their mean and `per_seed` gives the spread."""
    bt = pd.read_parquet(PROC / "bt_hourly.parquet", columns=["origin", "h", "pred_lp", "pred_aqi", "actual_aqi", "weather", "season"])
    bt = bt[(bt["weather"] == "forecast") & bt["season"].isin(SEASONS)]
    d = bt.merge(nn_df, on=["origin", "h"], how="inner")
    clip = lambda lp: aqi.pm_to_aqi(np.clip(np.expm1(lp), 0, None))  # noqa: E731
    d["nn_aqi"] = clip(d["nn_lp"].values)
    d["blend_aqi"] = clip(0.5 * (d["nn_lp"].values + d["pred_lp"].values))
    d["blend7_aqi"] = np.where(d["h"] >= BLEND_FROM_H, d["blend_aqi"], d["pred_aqi"])  # as in the shadow
    rows = []
    for (y, (lo, hi)) in [(y, b) for y in SEASONS for b in BUCKETS + [(1, 24), (1, 168)]]:
        g = d[(d["season"] == y) & (d["h"] >= lo) & (d["h"] <= hi)]
        if not len(g):
            continue
        rows.append({"season": y, "hours": f"{lo}-{hi}", "n": int(len(g)),
                     **{k: round(float((g[f"{k}_aqi"] - g["actual_aqi"]).abs().mean()), 2)
                        for k in ("pred", "nn", "blend", "blend7")}})
    for k, sd in enumerate(per_seed or []):
        e = bt.merge(sd, on=["origin", "h"], how="inner")
        e["a"] = clip(e["nn_lp"].values)
        for r in rows:
            lo, hi = map(int, r["hours"].split("-"))
            g = e[(e["season"] == r["season"]) & (e["h"] >= lo) & (e["h"] <= hi)]
            r[f"nn_seed{k}"] = round(float((g["a"] - g["actual_aqi"]).abs().mean()), 2)
    out = {"rows": rows, "pairs": int(len(d))}
    (PROC / "nn_test.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- shadow mode (real use, not on the site)
LOG = PROC.parent / "forecast_log"
SEEDS = (0, 1, 2)


def forecast(o: pd.Timestamp, wx_issue: pd.DataFrame) -> np.ndarray:
    """log PM2.5 for hours 1..HOR from origin `o`: retrained on everything before it, 3 seeds
    averaged; `wx_issue` is the weather the forecast is issued with (reanalysis + forecast)."""
    wx = load_weather()
    data = Data(o, weather_features(wx), splice_forecast_weather(wx, load_prev_runs_ensemble()),
                wx_live=weather_features(wx_issue))
    pos = np.array([int(data.idx.get_loc(o))])
    return np.mean([predict(train(data, seed=s), data, pos)["nn_lp"].values for s in SEEDS], axis=0)


def shadow(stem: str) -> dict | None:
    """The network's forecast and the blend next to an issued forecast (data/forecast_log/<stem>.json),
    from the same inputs: data up to its origin and the weather it was issued with
    (<stem>_wx.parquet). Retrained on everything before the origin, 3 seeds averaged.
    Written once to <stem>_nn.json; the site's forecast is not touched. `backfilled`: made more
    than 2 h after the forecast was issued (not in the same morning run)."""
    out_path = LOG / f"{stem}_nn.json"
    wx_path = LOG / f"{stem}_wx.parquet"
    if out_path.exists() or not wx_path.exists():
        return None
    js = json.loads((LOG / f"{stem}.json").read_text())
    o = pd.Timestamp(js["origin"])
    nn_lp = forecast(o, pd.read_parquet(wx_path))
    t = o + pd.to_timedelta(np.arange(1, HOR + 1), unit="h")
    # since the blend is served, the issued forecast keeps LightGBM's own hours next to it
    gbm = pd.Series({pd.Timestamp(r["t"]): r["aqi"] for r in js.get("hourly_7d_gbm", js["hourly_7d"])})
    gbm_lp = np.log1p(aqi.aqi_to_pm(gbm.reindex(t).values.astype(float)))
    now = pd.Timestamp.now(tz="Asia/Tashkent")
    res = nn_record(js["origin"], js["generated_at"], now.isoformat(), nn_lp, gbm_lp,
                    bool(now - pd.Timestamp(js["generated_at"]) > pd.Timedelta(hours=2)), list(SEEDS))
    from .safe_io import write_text
    write_text(out_path, json.dumps(res, ensure_ascii=False))
    return res


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="test", choices=["test", "shadow", "serve"])
    ap.add_argument("--backfill", action="store_true", help="shadow: also every earlier forecast that saved its weather")
    ap.add_argument("--origin", help="serve: the forecast origin")
    ap.add_argument("--wx", help="serve: parquet with the weather the forecast is issued with")
    ap.add_argument("--out", help="serve: where to write {origin, seeds, nn_lp}")
    a = ap.parse_args()
    if a.what == "serve":
        t0 = time.time()
        lp = forecast(pd.Timestamp(a.origin), pd.read_parquet(a.wx))
        from .safe_io import write_text
        write_text(Path(a.out), json.dumps({"origin": a.origin, "seeds": list(SEEDS), "nn_lp": [round(float(v), 5) for v in lp]}))
        print(f"network forecast from {a.origin}: {time.time() - t0:.0f}s", flush=True)
        return
    if a.what == "shadow":
        from .live import issued_stems  # the forecasts scored in real use: the first of each day
        stems = issued_stems()
        for st in (stems if a.backfill else stems[-1:]):
            t0 = time.time()
            if shadow(st):
                print(f"shadow {st}: written ({time.time() - t0:.0f}s)", flush=True)
        return
    runs = [run(seed=k) for k in (0, 1, 2)]
    mean = runs[0][["origin", "h"]].assign(nn_lp=np.mean([r["nn_lp"].values for r in runs], axis=0))
    res = compare(mean, runs)
    t = pd.DataFrame(res["rows"]).rename(columns={"pred": "lightgbm"})
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
