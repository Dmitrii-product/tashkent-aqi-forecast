"""US AQI <-> PM2.5 conversion (EPA breakpoints).

The model works in PM2.5 concentration (log space) because pollution physics is roughly
multiplicative there; AQI is piecewise-linear in concentration with kinks that would
otherwise distort the loss. Breakpoints are applied as a continuous piecewise-linear
map (the 0.1 µg/m³ gaps of the official table are closed) so the map is invertible.
"""
from __future__ import annotations

import numpy as np

# (concentration breakpoints, AQI breakpoints) — 2024 revision of the PM2.5 NAAQS table
BP_2024 = (
    np.array([0.0, 9.0, 35.4, 55.4, 125.4, 225.4, 325.4, 500.0]),
    np.array([0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 700.0]),
)
# pre-2024 table
BP_2012 = (
    np.array([0.0, 12.0, 35.4, 55.4, 150.4, 250.4, 350.4, 500.4]),
    np.array([0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0]),
)

BP = BP_2024

CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for sensitive groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very unhealthy", "#8f3f97"),
    (301, 10_000, "Hazardous", "#7e0023"),
]


def pm_to_aqi(pm, bp=None):
    c, a = bp or BP
    return np.interp(np.asarray(pm, dtype=float), c, a, right=np.nan)


def aqi_to_pm(aqi, bp=None):
    c, a = bp or BP
    return np.interp(np.asarray(aqi, dtype=float), a, c, right=np.nan)


def category_index(aqi) -> np.ndarray:
    aqi = np.asarray(aqi, dtype=float)
    edges = np.array([50.5, 100.5, 150.5, 200.5, 300.5])
    return np.searchsorted(edges, aqi)
