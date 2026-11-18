"""Hypothesis 4 of the actors check: is Sentinel-5P SO2 over the city (a trace of burning coal and
tyres) related to the pollution the weather does not explain? Rules fixed before the download:
docs/checks/2026-11-07-actors.md (hypothesis 4 and the note on it).

SO2 is read like NO2 in aqf/satellite.py: MEEO's public copy of the level-2 products on AWS, the
orbits passing Tashkent around 13:00 local, a 1.4°×1° window of the total vertical column
(mol/m², shown in µmol/m²), pixels with qa ≥ 50 (product guidance; qa ≥ 75 also kept), mean within
15 km of the centre and over a ring 30-50 km out. OFFL only, October-March days from 30.09.2020;
each day written once to data/satellite/so2/<YYYYMMDD>_OFFL.json (n = 0 when cloudy).

Test: Spearman ρ between yesterday's city SO2 and today's residual r = log(actual / forecast) of
aqf.stagnation (next-day forecast from 06:00 with the actual weather, model fixed per winter);
confirmed if ρ ≥ 0.1 with the 95% interval (bootstrap over 7-day blocks) above zero on winters
2024/25 and 2025/26 separately.

    PYTHONPATH=. .venv/bin/python research/so2_test.py fetch    # ≈15-20 min
    PYTHONPATH=. .venv/bin/python research/so2_test.py score    # -> data/processed/so2_test.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from aqf.data import PROC
from aqf.safe_io import write_text
from aqf.satellite import CITY_KM, DIR, PASS_UTC, RING_KM, _keys, _km, _window
from aqf.silam import _session

SO2_DIR = DIR / "so2"
ORBIT_RE = re.compile(r"S5P_(\w{4})_L2__SO2____(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{5})_")
COL = "PRODUCT_sulfurdioxide_total_vertical_column_4326.tif"
QA = "PRODUCT_qa_value_4326.tif"
QA_MAIN, QA_ALT = 50, 75
JUDGED = (2024, 2025)
N_BOOT, BLOCK = 2000, 7
rng = np.random.default_rng(0)


def days() -> list[pd.Timestamp]:
    d = pd.date_range("2020-09-30", "2026-11-02", freq="D")
    return [x for x in d if x.month in (9, 10, 11, 12, 1, 2, 3) and not (x.month == 9 and x.day < 30)]


def so2_day(session, day: pd.Timestamp) -> dict | None:
    keys = _keys(session, f"COGT/OFFL/L2__SO2___/{day:%Y/%m/%d}/")
    orbits = []
    for k in keys:
        m = ORBIT_RE.search(k)
        if m and k.endswith(COL):
            orbits.append((k, pd.Timestamp(m.group(2)), pd.Timestamp(m.group(3)), m.group(4)))
    if not orbits:
        return None
    lo, hi = day + pd.Timedelta(hours=PASS_UTC[0]), day + pd.Timedelta(hours=PASS_UTC[1])
    vals = {QA_MAIN: ([], []), QA_ALT: ([], [])}
    used = []
    for k, t0, t1, orbit in sorted(orbits, key=lambda o: o[1]):
        if t1 < lo or t0 > hi:
            continue
        a, lat, lon = _window(k)
        if not np.isfinite(a).any():
            continue
        q, _, _ = _window(k.replace(COL, QA))
        d = _km(lat, lon)
        v = a * 1e6  # mol/m² -> µmol/m²
        for qmin, (city, ring) in vals.items():
            ok = np.isfinite(v) & (q >= qmin) & (q <= 100) & (np.abs(v) < 50_000)
            city.append(v[ok & (d <= CITY_KM)])
            ring.append(v[ok & (d >= RING_KM[0]) & (d <= RING_KM[1])])
        used.append(orbit)
    out = {"date": f"{day:%Y-%m-%d}", "product": "OFFL", "unit": "µmol/m²", "orbits": used}
    for qmin, (city, ring) in vals.items():
        c, r = (np.concatenate(x) if x else np.array([]) for x in (city, ring))
        out[f"qa{qmin}"] = {"city_n": int(len(c)), "city": round(float(c.mean()), 1) if len(c) else None,
                            "ring_n": int(len(r)), "ring": round(float(r.mean()), 1) if len(r) else None}
    return out


def fetch(workers: int = 6) -> None:
    import rasterio
    SO2_DIR.mkdir(parents=True, exist_ok=True)
    todo = [d for d in days() if not (SO2_DIR / f"{d:%Y%m%d}_OFFL.json").exists()]
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
               GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="FALSE")
    t0, saved, failed = time.time(), 0, 0

    def one(day):
        with rasterio.Env(**env):
            try:
                res = so2_day(_session(), day)
            except Exception as e:  # noqa: BLE001
                return day, f"FAIL {type(e).__name__}: {e}"
        if res is None:
            return day, "none"
        path = SO2_DIR / f"{day:%Y%m%d}_OFFL.json"
        if not path.exists():
            write_text(path, json.dumps(res))
        return day, "saved"

    with ThreadPoolExecutor(workers) as ex:
        for i, (day, res) in enumerate(ex.map(one, todo), 1):
            saved += res == "saved"
            if res.startswith("FAIL"):
                failed += 1
                print(f"{day:%Y-%m-%d} {res}", flush=True)
            if i % 100 == 0:
                print(f"{i}/{len(todo)} days, {time.time() - t0:.0f}s", flush=True)
    print(f"SO2: {saved} saved, {failed} failed, {len(todo) - saved - failed} without orbits ({time.time() - t0:.0f}s)")


def load() -> pd.DataFrame:
    rows = []
    for p in sorted(SO2_DIR.glob("????????_OFFL.json")):
        j = json.loads(p.read_text())
        row = {"day": pd.Timestamp(j["date"])}
        for q in (QA_MAIN, QA_ALT):
            row[f"city{q}"], row[f"n{q}"] = j[f"qa{q}"]["city"], j[f"qa{q}"]["city_n"]
            row[f"ring{q}"] = j[f"qa{q}"]["ring"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("day").sort_index()


def spearman_boot(x: pd.Series, y: pd.Series) -> list[float]:
    d = pd.concat([x, y], axis=1, sort=True).dropna()
    rk = d.rank().values
    est = float(np.corrcoef(rk[:, 0], rk[:, 1])[0, 1])
    bl = [np.arange(i, min(i + BLOCK, len(d))) for i in range(0, len(d), BLOCK)]
    out = []
    for _ in range(N_BOOT):
        idx = np.concatenate([bl[i] for i in rng.integers(0, len(bl), len(bl))])
        s = d.iloc[idx].rank().values
        out.append(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
    lo, hi = np.percentile(out, [2.5, 97.5])
    return [round(est, 3), round(float(lo), 3), round(float(hi), 3), int(len(d))]


def score() -> dict:
    so2 = load()
    st = pd.read_parquet(PROC / "stagnation.parquet")
    st.index = pd.to_datetime(st.index).normalize()
    r = np.log(st["actual"] / st["forecast"])
    y_so2 = so2.shift(1, freq="D")  # yesterday's overpass for day D
    out = {"days_with_so2": {q: int(so2[f"n{q}"].gt(0).sum()) for q in (QA_MAIN, QA_ALT)}, "by_winter": {}}
    for season, g in st.groupby("season"):
        idx = g.index
        rec = {}
        for q in (QA_MAIN, QA_ALT):
            x = y_so2[f"city{q}"].where(y_so2[f"n{q}"] > 0).reindex(idx)
            rec[f"rho_qa{q}"] = spearman_boot(x, r.reindex(idx))
            rec[f"rho_qa{q}_city_minus_ring"] = spearman_boot((y_so2[f"city{q}"] - y_so2[f"ring{q}"]).reindex(idx), r.reindex(idx))
            rec[f"rho_qa{q}_vs_aqi_same_day"] = spearman_boot(so2[f"city{q}"].where(so2[f"n{q}"] > 0).reindex(idx), st["actual"])
        out["by_winter"][int(season)] = rec
    j = out["by_winter"]
    out["confirmed"] = bool(all(j[y][f"rho_qa{QA_MAIN}"][0] >= 0.1 and j[y][f"rho_qa{QA_MAIN}"][1] > 0 for y in JUDGED))
    (PROC / "so2_test.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch"]:
        fetch()
    else:
        print(json.dumps(score(), indent=1))
