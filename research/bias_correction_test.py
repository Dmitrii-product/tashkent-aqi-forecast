"""Would subtracting the recent bias of verified forecasts reduce the error? (rejected)

At every backtest origin the errors of forecasts already verified over the last W days are
known; their mean (in log PM2.5, overall or per lead day) is subtracted from the new forecast
with weight k. W in {7, 14, 28}, k in {0.25, 0.5, 0.75, 1}, overall or by lead: 24 variants,
scored on the realistic backtest (archived forecast weather) for the winters 2024/25, 2025/26
and October 2026.

Result (2026-10-26): no variant improved both winters and October 2026; at best -0.3..-0.7
AQI on one winter and worse on another, up to +6 AQI. Biases changed faster than the correction
caught up (early October 2026 the forecast ran 21-28 low, then turned high). The drift monitor
(aqf.live.drift) warns instead. Needs data/processed/bt_hourly.parquet (aqf.backtest).

    PYTHONPATH=. .venv/bin/python research/bias_correction_test.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aqf import aqi
from aqf.data import PROC


def main() -> pd.DataFrame:
    b = pd.read_parquet(PROC / "bt_hourly.parquet", columns=["origin", "h", "T", "pred_lp", "actual_aqi", "weather", "season"])
    b = b[b["weather"] == "forecast"].copy()
    b["e"] = b["pred_lp"] - np.log1p(aqi.aqi_to_pm(b["actual_aqi"].values))
    b["lead"] = np.ceil(b["h"] / 24).astype(int)
    b["d0"], b["dT"] = b["origin"].dt.normalize(), b["T"].dt.normalize()
    b["pred_aqi"] = aqi.pm_to_aqi(np.clip(np.expm1(b["pred_lp"]), 0, None))
    E = b.groupby(["dT", "lead"])["e"].agg(["sum", "count"]).unstack("lead").fillna(0)
    E = E.reindex(pd.date_range(E.index.min(), E.index.max() + pd.Timedelta(days=1), freq="D"), fill_value=0)
    S, N = E["sum"], E["count"]

    def bias_table(W: int, by_lead: bool) -> pd.DataFrame:
        s, n = S.rolling(W, min_periods=1).sum().shift(1), N.rolling(W, min_periods=1).sum().shift(1)
        if not by_lead:  # errors verified on days [d0-W, d0-1], all leads together
            tot = (s.sum(axis=1) / n.sum(axis=1).replace(0, np.nan)).where(n.sum(axis=1) >= 72)
            return pd.DataFrame({L: tot for L in S.columns})
        return (s / n.replace(0, np.nan)).where(n >= 72)

    def mae(df, col, lo, hi):
        s = df[(df["h"] >= lo) & (df["h"] <= hi)]
        return (s[col] - s["actual_aqi"]).abs().mean()

    rows = []
    key = pd.MultiIndex.from_arrays([b["d0"], b["lead"]])
    for W in (7, 14, 28):
        for by_lead in (False, True):
            bias = np.nan_to_num(bias_table(W, by_lead).stack().reindex(key).values)
            for k in (0.25, 0.5, 0.75, 1.0):
                b["corr"] = aqi.pm_to_aqi(np.clip(np.expm1(b["pred_lp"].values - k * bias), 0, None))
                for y in (2024, 2025, 2026):
                    s = b[b["season"] == y]
                    rows.append({"W": W, "by_lead": by_lead, "k": k, "season": y,
                                 **{f"d{lo}_{hi}": round(mae(s, "corr", lo, hi) - mae(s, "pred_aqi", lo, hi), 2)
                                    for lo, hi in ((1, 24), (25, 48), (1, 168))}})
    out = pd.DataFrame(rows)
    print("change in MAE (AQI, negative = better)")
    print(out.pivot_table(index=["W", "by_lead", "k"], columns="season", values=["d1_24", "d25_48", "d1_168"]).to_string())
    return out


if __name__ == "__main__":
    main()
