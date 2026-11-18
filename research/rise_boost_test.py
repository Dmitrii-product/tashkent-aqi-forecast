"""Raise the forecast when pollution is rising and the weather is stagnant? (owner, 2026-11-18)

Rules fixed before the run: docs/checks/2026-11-18-rise-boost.md. The 06:00 issue on day I; for
D = I (today) and D = I + 1 (tomorrow) the hourly PM2.5 forecast is multiplied by e^k when both hold:
  * rise: measured AQI over the 24 h before the issue >= 1.25 x the 24 h before that;
  * stagnant D: the archived forecast weather made before the issue (4-model mean; run 1 day ahead
    for D = I, 2 days ahead for D = I + 1) has daily mean 10 m wind <= the month's median (from those
    forecasts before 2025-10-01) and < 1 mm of rain.
k from {0, .05, .1, .15, .2, .3} by the 1-day daily MAE on 2024/25; passes if the 1-day daily MAE
falls on both 2025/26 and October 2026 (to 26.10) and the hourly MAE over hours 1-42 is not worse
by more than 0.1 on either. Forecast: backtest of model version 2026-11-04 (blend7 of
data/processed/nn_blend_v2026-10-28.parquet, archived forecast weather). Nothing is trained.

    PYTHONPATH=. .venv/bin/python research/rise_boost_test.py      # -> data/processed/rise_boost_test.json

Result (2026-11-18): rejected, k = 0 is best on 2024/25 (1-day MAE 18.63; k 0.05 18.66, 0.10 18.82).
At k = 0.10: 2025/26 19.29 -> 19.40, October 2026 19.55 -> 19.89. On the flagged days (18 / 20 / 3)
the forecast is not low already (bias +3.4 / -1.1 / +7.0): the model sees the rise and the stagnation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import PROC, load_prev_runs_ensemble, load_target
from aqf.live import LOG, issued_stems
from aqf.models import daily_aqi

KS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
RISE = 1.25
MED_UNTIL = "2025-10-01"
COLD = [10, 11, 12, 1, 2, 3]
SEASONS = {2024: "2024/25 (choice)", 2025: "2025/26", 2026: "Oct 2026 (to 26.10)"}
RNG = np.random.default_rng(0)


def rise_flags(A: pd.Series, origins: pd.DatetimeIndex) -> pd.Series:
    """Measured AQI over the last 24 h before each issue vs the 24 h before that (>= 18 hours each)."""
    out = {}
    for o in origins:
        last, prev = A[o - pd.Timedelta(hours=23): o].dropna(), A[o - pd.Timedelta(hours=47): o - pd.Timedelta(hours=24)].dropna()
        out[o] = bool(len(last) >= 18 and len(prev) >= 18 and last.mean() >= RISE * prev.mean())
    return pd.Series(out)


def stagnant_days(ens: dict) -> dict[int, pd.Series]:
    """Per lead 0 / 1: is day D stagnant in the forecast made before the issue (runs lead + 1 days ahead)."""
    w2 = ens[2]["wind_speed_10m"]
    dm = w2[w2.index < MED_UNTIL].resample("D").mean()
    med = dm.groupby(dm.index.month).median()
    out = {}
    for lead in (0, 1):
        f = ens[lead + 1]
        ws = f["wind_speed_10m"].resample("D").agg(["mean", "count"])
        rain = f["precipitation"].resample("D").sum()
        ok = (ws["count"] >= 20) & (ws["mean"] <= med.reindex(ws.index.month).values) & (rain < 1)
        out[lead] = ok
    return out


def apply(f: pd.DataFrame, k: float) -> np.ndarray:
    """Forecast AQI with PM2.5 x e^k on the flagged hours."""
    v = f["fc"].to_numpy(float).copy()
    m = f["boost"].to_numpy(bool)
    if k and m.any():
        v[m] = aqi.pm_to_aqi(aqi.aqi_to_pm(v[m]) * np.exp(k))
    return v


def frame(fc: pd.DataFrame, A: pd.Series, stag: dict, rise: pd.Series) -> pd.DataFrame:
    f = fc.copy()
    f["D"] = f["T"].dt.normalize()
    f["lead"] = (f["D"] - f["origin"].dt.normalize()).dt.days
    f["rise"] = f["origin"].map(rise).fillna(False).astype(bool)
    st = np.zeros(len(f), bool)
    for lead in (0, 1):
        m = (f["lead"] == lead).to_numpy()
        st[m] = stag[lead].reindex(f.loc[m, "D"]).fillna(False).to_numpy(bool)
    f["stag"] = st
    f["boost"] = f["rise"] & f["stag"] & f["lead"].isin([0, 1])
    if "act" not in f:
        f["act"] = A.reindex(f["T"]).to_numpy()
    return f


def metrics(f: pd.DataFrame, Ad: pd.Series, k: float) -> dict:
    f = f.assign(p=apply(f, k))
    d1 = f[f["lead"] == 1].groupby(["origin", "D"]).agg(p=("p", "mean"), n=("p", "size"), boost=("boost", "any")).reset_index()
    d1 = d1[d1["n"] == 24]
    d1["a"] = Ad.reindex(d1["D"]).to_numpy()
    d1 = d1.dropna(subset=["a"])
    h = f[(f["h"] >= 1) & (f["h"] <= 42)].dropna(subset=["act"])
    red = d1["a"] > 150
    return {"d1": d1, "mae_d1": float((d1["p"] - d1["a"]).abs().mean()), "mae_h42": float((h["p"] - h["act"]).abs().mean()),
            "n_days": int(len(d1)), "n_boost_days": int(d1["boost"].sum()),
            "bias_boost_days": float((d1.loc[d1["boost"], "p"] - d1.loc[d1["boost"], "a"]).mean()) if d1["boost"].any() else None,
            "red_days": int(red.sum()), "red_caught": int((red & (d1["p"] > 150)).sum()),
            "false_alarms": int((~red & (d1["p"] > 150)).sum())}


def block_ci(d0: pd.DataFrame, dk: pd.DataFrame, draws: int = 2000) -> list[float]:
    """95% interval of the change in 1-day MAE, bootstrap over 7-day blocks of issue days."""
    e = pd.DataFrame({"e0": (d0["p"] - d0["a"]).abs().to_numpy(), "ek": (dk["p"] - dk["a"]).abs().to_numpy()},
                     index=d0["origin"].to_numpy())
    blocks = [g for _, g in e.groupby((e.index - e.index.min()).days // 7)]
    diffs = []
    for _ in range(draws):
        s = pd.concat([blocks[i] for i in RNG.integers(0, len(blocks), len(blocks))])
        diffs.append(s["ek"].mean() - s["e0"].mean())
    return [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)]


def live_frame(A: pd.Series, stag: dict) -> pd.DataFrame:
    """Issued forecasts from 4 Nov 2026 (model version 2026-11-04 on), as served."""
    rows = []
    for stem in issued_stems():
        if stem < "2026110400":
            continue
        js = json.loads((LOG / f"{stem}.json").read_text())
        o = pd.Timestamp(js["origin"])
        for r in js["hourly_7d"]:
            t = pd.Timestamp(r["t"])
            rows.append({"origin": o, "T": t, "h": int((t - o) / pd.Timedelta(hours=1)), "fc": float(r["aqi"])})
    fc = pd.DataFrame(rows)
    return frame(fc, A, stag, rise_flags(A, pd.DatetimeIndex(fc["origin"].unique())))


def main() -> dict:
    pm = load_target()[0]["pm"]
    A = pd.Series(aqi.pm_to_aqi(pm.to_numpy()), index=pm.index)
    Ad = daily_aqi(pm)
    stag = stagnant_days(load_prev_runs_ensemble())
    b = pd.read_parquet(PROC / "nn_blend_v2026-10-28.parquet", columns=["origin", "h", "season", "blend7", "actual"])
    b = b[(b["origin"].dt.hour == 6) & b["origin"].dt.month.isin(COLD)]
    b = b.rename(columns={"blend7": "fc", "actual": "act"})
    b["T"] = b["origin"] + pd.to_timedelta(b["h"], unit="h")
    f = frame(b, A, stag, rise_flags(A, pd.DatetimeIndex(b["origin"].unique())))

    out = {"rules": "docs/checks/2026-11-18-rise-boost.md", "seasons": {}}
    for season, name in SEASONS.items():
        s = f[f["season"] == season]
        out["seasons"][name] = {f"k={k}": {kk: v for kk, v in metrics(s, Ad, k).items() if kk != "d1"} for k in KS}
    choice = out["seasons"][SEASONS[2024]]
    k_best = min(KS, key=lambda k: choice[f"k={k}"]["mae_d1"])
    out["k_chosen"] = k_best
    passes = k_best > 0
    for season in (2025, 2026):
        r = out["seasons"][SEASONS[season]]
        passes &= r[f"k={k_best}"]["mae_d1"] < r["k=0.0"]["mae_d1"] and r[f"k={k_best}"]["mae_h42"] <= r["k=0.0"]["mae_h42"] + 0.1
    out["passes"] = bool(passes)
    k_show = k_best if k_best > 0 else 0.10  # the reference intervals use k = 0.10 if nothing was chosen
    out["ci_k"] = k_show
    for season, name in SEASONS.items():
        s = f[f["season"] == season]
        out["seasons"][name]["ci_mae_d1_change"] = block_ci(metrics(s, Ad, 0)["d1"], metrics(s, Ad, k_show)["d1"])

    lf = live_frame(A, stag)
    out["live_4_17_nov"] = {f"k={k}": {kk: v for kk, v in metrics(lf, Ad, k).items() if kk != "d1"} for k in (0.0, k_show)}
    out["live_flags"] = {f"{o:%Y-%m-%d}": {"rise": bool(g["rise"].iloc[0]),
                                           "stag_today": bool(g.loc[g["lead"] == 0, "stag"].any()),
                                           "stag_tomorrow": bool(g.loc[g["lead"] == 1, "stag"].any())}
                         for o, g in lf.groupby("origin")}
    (PROC / "rise_boost_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    r = main()
    print("k chosen on 2024/25:", r["k_chosen"], "| passes:", r["passes"])
    for name, s in r["seasons"].items():
        print(f"\n{name}: 1-day MAE change 95% CI at k={r['ci_k']}: {s['ci_mae_d1_change']}")
        for k in KS:
            m = s[f"k={k}"]
            print(f"  k={k:<4} 1-day MAE {m['mae_d1']:.2f}  h1-42 MAE {m['mae_h42']:.2f}  days {m['n_days']} boosted {m['n_boost_days']}"
                  f"  bias on boosted {m['bias_boost_days'] if m['bias_boost_days'] is None else round(m['bias_boost_days'], 1)}"
                  f"  red {m['red_caught']}/{m['red_days']} false {m['false_alarms']}")
    print("\nlive 4-17 Nov 2026:", json.dumps(r["live_4_17_nov"], indent=0))
    print("flags:", r["live_flags"])
