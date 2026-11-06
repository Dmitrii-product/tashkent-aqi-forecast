"""Satellite signals around Tashkent, kept every day and set against the measured air: NO2 from
Sentinel-5P/TROPOMI (burning: heating, traffic, power plants) and fires from NASA FIRMS (VIIRS).
Neither is a model input; they are watched next to the fact on their own satellites tab.

NO2. MEEO's public copy of the Sentinel-5P level-2 products on AWS (s3://meeo-s5p, no account)
has every orbit as a cloud-optimised GeoTIFF on a 0.035° grid: tropospheric NO2 column (mol/m²)
and qa_value (×100). Tashkent is seen once a day, about 08:00 UTC (13:00 local), by one or two
orbits. A 1.4°×1° window is read from the orbits passing then (a few hundred kB), pixels with
qa ≥ 75 are kept, and the day gets the mean over the city (within 15 km of the centre) and over a
ring 30–50 km out (background), in µmol/m². OFFL (final, about 9 days later) goes back to May
2018; NRTI (same day) covers the latest days until OFFL arrives. Each (day, product) is written
once, data/satellite/no2/<YYYYMMDD>_<OFFL|NRTI>.json, only after the day's later orbits are in the
bucket; a cloudy day is stored with n = 0 (that is a result, not a gap).

Fires. FIRMS publishes the last 7 days of VIIRS (Suomi NPP) detections openly and whole years in
per-country archives the following year (2018–2025 now; January–October 2026 is not public yet).
Detections within 200 km are kept: each complete day of the 7-day file once
(data/satellite/firms/<YYYYMMDD>.parquet, also when empty — no fires is a result), the yearly
archives of Uzbekistan, Kazakhstan, Tajikistan and Kyrgyzstan once per year (archive_<year>.parquet).
Industrial heat sources (the archive's type 2 and places that glow on 30+ days, such as the Almalyk
works) are counted apart from fires.

    .venv/bin/python -m aqf.satellite fetch          # new days (run_daily.sh)
    .venv/bin/python -m aqf.satellite fetch --all    # NO2 back to 2018 (once, ~40 min)
    .venv/bin/python -m aqf.satellite score
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .data import PROC, ROOT
from .safe_io import to_parquet, write_text
from .silam import _session

LAT, LON = 41.31, 69.28
DIR = ROOT / "data" / "satellite"
NO2_DIR, FIRMS_DIR = DIR / "no2", DIR / "firms"
SCORE = PROC / "satellite_score.json"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

S5P = "https://meeo-s5p.s3.amazonaws.com"
NO2_FROM = pd.Timestamp("2018-05-01")
CITY_KM, RING_KM = 15, (30, 50)
QA_MIN = 75
PASS_UTC = (7.0, 10.5)  # orbits whose time span meets this window (hours UTC) can see Tashkent
ORBIT_RE = re.compile(r"S5P_(\w{4})_L2__NO2____(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{5})_")
TROP = "PRODUCT_nitrogendioxide_tropospheric_column_4326.tif"

FIRMS = "https://firms.modaps.eosdis.nasa.gov/data"
NRT_FILE = f"{FIRMS}/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_Russia_Asia_7d.csv"
COUNTRIES = ("Uzbekistan", "Kazakhstan", "Tajikistan", "Kyrgyzstan")
ARCHIVE_FROM = 2018
NEAR_KM, CLOSE_KM = 200, 50
STATIC_DAYS = 30  # a 0.01° cell detected on this many days of the archive is a heat source, not a fire


class Invalid(Exception):
    pass


def _km(lat, lon) -> np.ndarray:
    la, lo = np.radians(LAT), np.radians(LON)
    p, q = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return 6371 * 2 * np.arcsin(np.sqrt(np.sin((p - la) / 2) ** 2 + np.cos(la) * np.cos(p) * np.sin((q - lo) / 2) ** 2))


# ---------------------------------------------------------------- NO2 (Sentinel-5P)
def _keys(session, prefix: str) -> list[str]:
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix} | ({"continuation-token": token} if token else {})
        r = session.get(S5P, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        keys += [c.findtext("s3:Key", namespaces=NS) for c in root.findall("s3:Contents", NS)]
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        if not token:
            return keys


def _window(key: str):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(f"/vsicurl/{S5P}/{key}") as src:
        w = from_bounds(LON - 0.7, LAT - 0.5, LON + 0.7, LAT + 0.5, src.transform)
        a = src.read(1, window=w, boundless=True, fill_value=src.nodata if src.nodata is not None else np.nan).astype(float)
        t = src.window_transform(w)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    rows, cols = np.indices(a.shape)
    lon = t.c + (cols + 0.5) * t.a
    lat = t.f + (rows + 0.5) * t.e
    return a, lat, lon


def no2_day(session, day: pd.Timestamp, product: str) -> dict | None:
    """The day's NO2 over Tashkent, or None while the bucket does not have the whole day yet."""
    keys = _keys(session, f"COGT/{product}/L2__NO2___/{day:%Y/%m/%d}/")
    orbits = []
    for k in keys:
        m = ORBIT_RE.search(k)
        if m and k.endswith(TROP):
            t0, t1 = (pd.Timestamp(m.group(i)) for i in (2, 3))
            orbits.append((k, t0, t1, m.group(4)))
    # wait until the bucket has the day's later orbits too (NRTI arrives orbit by orbit)
    if not orbits or max(t0 for _, t0, _, _ in orbits) < day + pd.Timedelta(hours=12):
        return None
    lo, hi = day + pd.Timedelta(hours=PASS_UTC[0]), day + pd.Timedelta(hours=PASS_UTC[1])
    city_v, ring_v, out = [], [], []
    for k, t0, t1, orbit in sorted(orbits, key=lambda o: o[1]):
        if t1 < lo or t0 > hi:
            continue
        a, lat, lon = _window(k)
        if not np.isfinite(a).any():
            continue
        q, _, _ = _window(k.replace(TROP, "PRODUCT_qa_value_4326.tif"))
        ok = np.isfinite(a) & (q >= QA_MIN) & (q <= 100)
        v = a[ok] * 1e6  # mol/m² -> µmol/m²
        if len(v) and not ((v > -200) & (v < 3000)).all():
            raise Invalid(f"orbit {orbit}: NO2 outside -200..3000 µmol/m² ({v.min():.0f}..{v.max():.0f})")
        d = _km(lat, lon)[ok]
        c, r = v[d <= CITY_KM], v[(d >= RING_KM[0]) & (d <= RING_KM[1])]
        city_v.append(c)
        ring_v.append(r)
        out.append({"orbit": orbit, "start": f"{t0:%H:%M}", "end": f"{t1:%H:%M}", "city_n": int(len(c)),
                    "city": round(float(c.mean()), 1) if len(c) else None, "ring_n": int(len(r)),
                    "ring": round(float(r.mean()), 1) if len(r) else None})
    c, r = (np.concatenate(x) if x else np.array([]) for x in (city_v, ring_v))
    return {"date": f"{day:%Y-%m-%d}", "product": product, "unit": "µmol/m²", "qa_min": QA_MIN,
            "city_km": CITY_KM, "ring_km": list(RING_KM), "orbits": out,
            "city_n": int(len(c)), "city": round(float(c.mean()), 1) if len(c) else None,
            "city_median": round(float(np.median(c)), 1) if len(c) else None,
            "ring_n": int(len(r)), "ring": round(float(r.mean()), 1) if len(r) else None,
            "fetched_at": pd.Timestamp.now(tz="Asia/Tashkent").isoformat(timespec="seconds")}


def _no2_one(session, day: pd.Timestamp, product: str) -> str:
    path = NO2_DIR / f"{day:%Y%m%d}_{product}.json"
    if path.exists():  # written once, never replaced
        return "have"
    try:
        res = no2_day(session, day, product)
    except Invalid as e:
        return f"WARNING: NO2 {product} {day:%Y-%m-%d} not saved: {e}"
    except Exception as e:  # noqa: BLE001 - network or format trouble: try again next run
        return f"WARNING: NO2 {product} {day:%Y-%m-%d} not saved: {type(e).__name__}: {e}"
    if res is None:
        return "pending"
    if not path.exists():
        write_text(path, json.dumps(res, ensure_ascii=False))
    return "saved"


def fetch_no2(all_days: bool = False, workers: int = 6) -> int:
    """OFFL for missing days (back to 2018 with all_days, else the last 40 days) and NRTI for the
    last 12 days that have no OFFL yet. Returns the number of days that failed."""
    import rasterio
    NO2_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    have = {p.stem for p in NO2_DIR.glob("????????_*.json")}
    start = NO2_FROM if all_days else today - pd.Timedelta(days=40)
    todo = [(d, "OFFL") for d in pd.date_range(start, today - pd.Timedelta(days=1)) if f"{d:%Y%m%d}_OFFL" not in have]
    todo += [(d, "NRTI") for d in pd.date_range(today - pd.Timedelta(days=12), today)
             if f"{d:%Y%m%d}_OFFL" not in have and f"{d:%Y%m%d}_NRTI" not in have]
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
               GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="FALSE")
    t0, failed, saved = time.time(), 0, 0

    def one(task):
        with rasterio.Env(**env):
            return task, _no2_one(_session(), *task)

    with ThreadPoolExecutor(workers) as ex:
        for (d, prod), res in ex.map(one, todo):
            if res.startswith("WARNING"):
                print(res, flush=True)
                failed += 1
            elif res == "saved":
                saved += 1
    print(f"NO2: {saved} day(s) saved, {failed} failed ({time.time() - t0:.0f}s)", flush=True)
    return failed


# ---------------------------------------------------------------- fires (FIRMS)
def _near(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["km"] = _km(df["latitude"], df["longitude"])
    df = df[df["km"] <= NEAR_KM]
    conf = df["confidence"].astype(str).str[0].str.lower()  # n/h/l in the archive, nominal/high/low in NRT
    df["conf"] = conf
    keep = ["latitude", "longitude", "km", "acq_date", "acq_time", "frp", "conf", "daynight"] + (["type"] if "type" in df else [])
    return df[keep].reset_index(drop=True)


def fetch_firms() -> int:
    """Complete days of the 7-day file, and yearly archives not stored yet. Returns failures."""
    s = _session()
    FIRMS_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    try:
        r = s.get(NRT_FILE, timeout=180)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        need = {"latitude", "longitude", "acq_date", "acq_time", "frp", "confidence", "daynight"}
        if not need <= set(df.columns) or len(df) < 1000:
            raise Invalid(f"unexpected 7-day file ({len(df)} rows, columns {sorted(df.columns)[:6]}…)")
        dates = sorted(df["acq_date"].unique())
        if len(dates) < 5:
            raise Invalid(f"7-day file spans {len(dates)} days only")
        near = _near(df)
        today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        n = 0
        for d in dates[1:]:  # the first day of the window may be cut at its start
            if pd.Timestamp(d) > today - pd.Timedelta(days=2):  # night passes and late granules still coming
                continue
            path = FIRMS_DIR / f"{pd.Timestamp(d):%Y%m%d}.parquet"
            if not path.exists():
                to_parquet(near[near["acq_date"] == d].reset_index(drop=True), path)
                n += 1
        print(f"FIRMS: {n} day(s) saved from the 7-day file ({len(near)} detections within {NEAR_KM} km)", flush=True)
    except Exception as e:  # noqa: BLE001 - the 7-day window gives five more tries
        print(f"WARNING: FIRMS 7-day file not saved: {type(e).__name__}: {e}", flush=True)
        failed += 1
    for year in range(ARCHIVE_FROM, pd.Timestamp.now().year):
        path = FIRMS_DIR / f"archive_{year}.parquet"
        if path.exists():
            continue
        parts = []
        try:
            for c in COUNTRIES:
                r = s.get(f"{FIRMS}/country/viirs-snpp/{year}/viirs-snpp_{year}_{c}.csv", timeout=600)
                if r.status_code == 404:
                    raise FileNotFoundError(f"{c} {year} not published yet")
                r.raise_for_status()
                parts.append(_near(pd.read_csv(io.BytesIO(r.content))))
            a = pd.concat(parts, ignore_index=True).drop_duplicates(["latitude", "longitude", "acq_date", "acq_time"])
            yrs = pd.to_datetime(a["acq_date"]).dt.year
            if a.empty or (yrs != year).any():
                raise Invalid(f"archive {year}: {len(a)} rows, years {sorted(yrs.unique())}")
            to_parquet(a, path)
            print(f"FIRMS: archive {year} saved ({len(a)} detections within {NEAR_KM} km)", flush=True)
        except FileNotFoundError:
            continue
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: FIRMS archive {year} not saved: {type(e).__name__}: {e}", flush=True)
            failed += 1
    return failed


def fetch(all_days: bool = False) -> int:
    return fetch_no2(all_days) + fetch_firms()


# ---------------------------------------------------------------- daily series and correlations
def no2_series() -> pd.DataFrame:
    rows = {}
    for p in sorted(NO2_DIR.glob("????????_*.json")):
        js = json.loads(p.read_text())
        d = pd.Timestamp(js["date"])
        if js["product"] == "OFFL" or d not in rows:  # final product wins over the quick one
            rows[d] = js
    df = pd.DataFrame([{"date": d, "no2": js["city"], "no2_ring": js["ring"], "no2_n": js["city_n"], "product": js["product"]}
                       for d, js in rows.items()])
    return df.set_index("date").sort_index() if len(df) else df


def fire_series() -> pd.DataFrame:
    """Fires per day within 200 and 50 km (heat sources apart); days with no data are absent."""
    arch = [pd.read_parquet(p) for p in sorted(FIRMS_DIR.glob("archive_*.parquet"))]
    nrt = {p.stem: pd.read_parquet(p) for p in sorted(FIRMS_DIR.glob("????????.parquet"))}
    a = pd.concat(arch, ignore_index=True) if arch else pd.DataFrame()
    cell = lambda df: list(zip((df["latitude"] * 100).round().astype(int), (df["longitude"] * 100).round().astype(int)))  # noqa: E731
    static = set()
    if len(a):
        a["cell"] = cell(a)
        static |= set(a.loc[a.get("type", 0) == 2, "cell"])
        days = a.groupby("cell")["acq_date"].nunique()
        static |= set(days[days >= STATIC_DAYS].index)
    frames = [a] + [df.assign(acq_date=f"{pd.Timestamp(k):%Y-%m-%d}") for k, df in nrt.items()]
    all_ = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    covered = set(pd.to_datetime(a["acq_date"]).dt.normalize()) if len(a) else set()
    for y in {int(p.stem[-4:]) for p in FIRMS_DIR.glob("archive_*.parquet")}:  # an archive year covers every day of it
        covered |= set(pd.date_range(f"{y}-01-01", f"{y}-12-31"))
    covered |= {pd.Timestamp(k) for k in nrt}
    out = pd.DataFrame(index=pd.DatetimeIndex(sorted(covered), name="date"))
    out["fires"] = out["fires50"] = out["frp"] = out["heat"] = 0.0
    if len(all_):
        all_["cell"] = cell(all_)
        all_ = all_[all_["conf"] != "l"]
        all_["date"] = pd.to_datetime(all_["acq_date"]).dt.normalize()
        is_static = all_["cell"].isin(static)
        f, h = all_[~is_static], all_[is_static]
        out["fires"] = f.groupby("date").size().reindex(out.index, fill_value=0).astype(float)
        out["fires50"] = f[f["km"] <= CLOSE_KM].groupby("date").size().reindex(out.index, fill_value=0).astype(float)
        out["frp"] = f.groupby("date")["frp"].sum().reindex(out.index, fill_value=0).round(1)
        out["heat"] = h.groupby("date").size().reindex(out.index, fill_value=0).astype(float)
    return out


def _corr(x: pd.Series, y: pd.Series) -> dict | None:
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 20:
        return None
    return {"r": round(float(d.iloc[:, 0].rank().corr(d.iloc[:, 1].rank())), 2), "n": int(len(d))}


def score() -> dict:
    from .data import load_target
    pm = load_target()[0]["pm"]
    from . import aqi
    a = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    g = a.resample("D")
    act = g.mean().where(g.count() >= 18)
    noon = a[a.index.hour.isin([13, 14])].resample("D").mean()  # the satellite passes about 13:00 local
    no2, fire = no2_series(), fire_series()
    if not len(no2) and not len(fire):
        return {}
    df = pd.DataFrame({"aqi": act, "aqi13": noon})
    df = df.join(no2[["no2", "no2_ring", "no2_n"]] if len(no2) else None, how="outer") if len(no2) else df
    df = df.join(fire, how="outer") if len(fire) else df
    df["aqi_next"] = df["aqi"].shift(-1, freq="D").reindex(df.index)
    end = df.index.max()
    sets = {"all": df.index == df.index, "winter": df.index.month.isin([11, 12, 1, 2]),
            "autumn": df.index.month.isin([9, 10]), "last90": df.index > end - pd.Timedelta(days=90)}
    pairs = {"no2_aqi": ("no2", "aqi"), "no2_aqi13": ("no2", "aqi13"), "no2_ring_aqi": ("no2_ring", "aqi"),
             "fires_aqi": ("fires", "aqi"), "fires_next": ("fires", "aqi_next")}
    corr = {k: {s: _corr(df.loc[m, x], df.loc[m, y]) for s, m in sets.items()} for k, (x, y) in pairs.items()}
    # how the last 30 days compare with the same calendar days of earlier years
    last = df.index[df["no2"].notna()].max() if "no2" in df and df["no2"].notna().any() else end
    years = []
    for y in range(last.year, ARCHIVE_FROM - 1, -1):
        t1 = last - pd.DateOffset(years=last.year - y)
        w = df[(df.index > t1 - pd.Timedelta(days=30)) & (df.index <= t1)]
        if len(w):
            years.append({"year": y, "to": f"{t1:%Y-%m-%d}",
                          "no2": round(float(w["no2"].mean()), 1) if "no2" in w and w["no2"].notna().any() else None,
                          "no2_days": int(w["no2"].notna().sum()) if "no2" in w else 0,
                          "fires": int(w["fires"].sum()) if "fires" in w and w["fires"].notna().any() else None,
                          "aqi": round(float(w["aqi"].mean())) if w["aqi"].notna().any() else None})
    # days between the last yearly fire archive and the first day of the 7-day file have no fire data
    arch_years = sorted(int(p.stem[-4:]) for p in FIRMS_DIR.glob("archive_*.parquet"))
    nrt_days = sorted(p.stem for p in FIRMS_DIR.glob("????????.parquet"))
    gap = None
    if arch_years and nrt_days:
        a, b = pd.Timestamp(f"{arch_years[-1] + 1}-01-01"), pd.Timestamp(nrt_days[0]) - pd.Timedelta(days=1)
        if b >= a:
            gap = {"from": f"{a:%Y-%m-%d}", "to": f"{b:%Y-%m-%d}", "daily_from": f"{pd.Timestamp(nrt_days[0]):%Y-%m-%d}"}
    recent = df[df.index > end - pd.Timedelta(days=120)]
    out = {"no2_from": f"{no2.index.min():%Y-%m-%d}" if len(no2) else None, "no2_to": f"{no2.index.max():%Y-%m-%d}" if len(no2) else None,
           "no2_days": int(no2["no2"].notna().sum()) if len(no2) else 0,
           "fire_days": int(len(fire)), "fire_gap": gap,
           "corr": corr, "years": years,
           "days": [{"date": f"{d:%Y-%m-%d}", "aqi": None if pd.isna(r.aqi) else round(r.aqi),
                     "no2": None if pd.isna(r.get("no2")) else round(r["no2"], 1),
                     "fires": None if pd.isna(r.get("fires")) else int(r["fires"])} for d, r in recent.iterrows()]}
    write_text(SCORE, json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "score"
    if what == "fetch":
        failed = fetch(all_days="--all" in sys.argv)
        if failed:
            print(f"WARNING: satellite fetch: {failed} item(s) not saved, retried next run")
    elif what == "score":
        print(json.dumps({k: v for k, v in score().items() if k != "days"}, ensure_ascii=False, indent=1))
    else:
        raise SystemExit("usage: python -m aqf.satellite fetch [--all] | score")
