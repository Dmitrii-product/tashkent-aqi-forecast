"""SILAM (FMI) air-quality forecasts for Tashkent: copied every day, kept, and scored against our
forecasts and what happened.

FMI publishes its global SILAM forecast (20 km grid, hourly, 5 days, CC BY 4.0) on a public S3
bucket that keeps only the last ~32 runs. Uzbekistan's National Committee on Ecology uses SILAM,
so it is the natural outside benchmark. Each run is copied as soon as we see it: PM2.5 and PM10
at the Tashkent grid point and its 3×3 neighbourhood, read straight from the remote HDF5 files
with HTTP range requests (about a third of each 34 MB file is read).

Safety. A run is written once (data/silam/<YYYYMMDD>.parquet, atomically) and only after it
passes every check: all 10 files present, 120 consecutive hourly steps starting on the run day,
the grid point at Tashkent, values finite and within 0–3000 µg/m³, PM10 ≥ PM2.5, not constant.
A run that fails is skipped with a warning and tried again on the next day while FMI still has
it. A saved run is never rewritten, so an outage or a bad upload later cannot touch it.

Fair comparison. Each of our issued forecasts is set against the SILAM run of the same day: SILAM
starts from 05:00 Tashkent time (its first hourly value ends at 06:00) and is uploaded at about
10:00, ours starts from 06:00 and is issued at 07:14 — the same starting point, give or take an
hour, so neither side gets a fresher look at the weather.

SILAM with a correction (tracked since 2026-10-27). SILAM starts every run from its own previous
forecast and runs 2–2.4 times below the measured PM2.5 (research/silam_bias_check.py). The
corrected forecast is the same run multiplied by measured / SILAM over the 7 days up to 06:00 of
its run day (each past hour taken from the run of its own day), i.e. only what was measured
when our forecast came out. It learns the current level from the measurements; the ups and downs
stay SILAM's.

    .venv/bin/python -m aqf.silam fetch    # every run in the bucket we do not have yet
    .venv/bin/python -m aqf.silam score    # SILAM vs our forecast vs fact
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import aqi
from .data import PROC, ROOT
from .safe_io import to_parquet, write_text

BUCKET = "https://fmi-opendata-silam-surface-netcdf.s3.eu-west-1.amazonaws.com"
DIR = ROOT / "data" / "silam"
SCORE = PROC / "silam_score.json"
LAT, LON = 41.31, 69.28  # Tashkent, city centre
PARAMS = ("PM25", "PM10")
DAYS = range(5)  # d0..d4: the run day and the four after it, 24 hourly steps each
UTC_OFFSET = pd.Timedelta(hours=5)  # stored in Tashkent time, as everything else here
KEY_RE = re.compile(r"global/(\d{8})/silam_glob_(v[\d_]+)_(\d{8})_(PM25|PM10)_d(\d)\.nc$")
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


# ---------------------------------------------------------------- remote files
def _session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                                                      allowed_methods=("GET", "HEAD"))))
    return s


class _Remote(io.RawIOBase):
    """Read-only seekable view of a remote file over HTTP range requests, for h5py. Small reads
    (HDF5 metadata) go through a block cache; large ones (compressed data chunks) are fetched as is."""

    BLOCK = 64 * 1024

    def __init__(self, url: str, size: int, session):
        super().__init__()
        self.url, self.size, self.s = url, size, session
        self.pos, self.cache, self.fetched = 0, {}, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def _get(self, a: int, b: int) -> bytes:
        r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=90)
        if r.status_code != 206 or len(r.content) != b - a:
            raise OSError(f"{self.url} bytes {a}-{b}: HTTP {r.status_code}, {len(r.content)} bytes")
        self.fetched += b - a
        return r.content

    def _block(self, k: int) -> bytes:
        if k not in self.cache:
            self.cache[k] = self._get(k * self.BLOCK, min((k + 1) * self.BLOCK, self.size))
        return self.cache[k]

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        a, b = self.pos, self.pos + n
        if n >= self.BLOCK:
            data = self._get(a, b)
        else:
            k0, k1 = a // self.BLOCK, (b - 1) // self.BLOCK
            data = b"".join(self._block(k) for k in range(k0, k1 + 1))[a - k0 * self.BLOCK:b - k0 * self.BLOCK]
        buf[:n] = data
        self.pos = b
        return n


def _list(session, prefix: str, delimiter: str | None = None) -> ET.Element:
    q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"} | ({"delimiter": delimiter} if delimiter else {})
    r = session.get(BUCKET + "/", params=q, timeout=60)
    r.raise_for_status()
    return ET.fromstring(r.content)


def runs_available(session) -> list[str]:
    root = _list(session, "global/", "/")
    return sorted(m.group(1) for p in root.iterfind("s3:CommonPrefixes/s3:Prefix", NS)
                  if (m := re.fullmatch(r"global/(\d{8})/", p.text or "")))


class Invalid(Exception):
    """A run that must not be saved (missing files or implausible data)."""


def _read_point(session, url: str, size: int, param: str) -> tuple[pd.DatetimeIndex, np.ndarray, float, float]:
    import h5py
    f = _Remote(url, size, session)
    with h5py.File(f, "r") as h:
        units = h[param].attrs.get("units", b"")
        units = units.decode() if isinstance(units, bytes) else str(units)
        if units != "ug/m3":
            raise Invalid(f"{param}: units {units!r}, expected ug/m3")
        lat, lon = h["lat"][:], h["lon"][:]
        i, j = int(np.abs(lat - LAT).argmin()), int(np.abs(lon - LON).argmin())
        if abs(lat[i] - LAT) > 0.15 or abs(lon[j] - LON) > 0.15 or not (0 < i < len(lat) - 1 and 0 < j < len(lon) - 1):
            raise Invalid(f"{param}: nearest grid point {lat[i]:.2f}, {lon[j]:.2f} is not Tashkent")
        block = h[param][:, i - 1:i + 2, j - 1:j + 2].astype(float)
        tu = h["time"].attrs["units"]
        tu = tu.decode() if isinstance(tu, bytes) else str(tu)
        m = re.fullmatch(r"seconds since (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)( UTC)?", tu.strip())
        if not m:
            raise Invalid(f"{param}: time units {tu!r}")
        t = pd.Timestamp(m.group(1)) + pd.to_timedelta(h["time"][:].astype("int64"), unit="s")
    return pd.DatetimeIndex(t), block, float(lat[i]), float(lon[j])


def fetch_run(session, run: str) -> pd.DataFrame:
    """One run as a frame (Tashkent time index), or Invalid. Nothing is written here."""
    root = _list(session, f"global/{run}/")
    files = {}
    for c in root.iterfind("s3:Contents", NS):
        key = c.findtext("s3:Key", namespaces=NS) or ""
        m = KEY_RE.search(key)
        if m and m.group(1) == run and m.group(3) == run:
            files[(m.group(4), int(m.group(5)))] = (key, int(c.findtext("s3:Size", namespaces=NS)), m.group(2),
                                                     c.findtext("s3:LastModified", namespaces=NS))
    want = [(p, d) for p in PARAMS for d in DAYS]
    missing = [f"{p}_d{d}" for p, d in want if (p, d) not in files]
    if missing:
        raise Invalid(f"incomplete ({len(want) - len(missing)}/{len(want)} files; missing {', '.join(missing)})")
    cols, fetched = {}, 0
    for p in PARAMS:
        ts, vals = [], []
        for d in DAYS:
            key, size, _, _ = files[(p, d)]
            t, block, la, lo = _read_point(session, f"{BUCKET}/{key}", size, p)
            if len(t) != 24:
                raise Invalid(f"{p}_d{d}: {len(t)} time steps, expected 24")
            ts.append(t)
            vals.append(block)
        t = ts[0].append(ts[1:])
        v = np.concatenate(vals)
        if not (t[1:] - t[:-1] == pd.Timedelta(hours=1)).all():
            raise Invalid(f"{p}: time steps are not consecutive hours")
        start = pd.Timestamp(run)
        if not (start <= t[0] <= start + pd.Timedelta(hours=1)):
            raise Invalid(f"{p}: starts {t[0]}, expected the run day {run}")
        if not np.isfinite(v).all() or (v < 0).any() or (v > 3000).any():
            raise Invalid(f"{p}: values missing or outside 0–3000 µg/m³ (min {np.nanmin(v):.3g}, max {np.nanmax(v):.3g})")
        if np.ptp(v[:, 1, 1]) == 0:
            raise Invalid(f"{p}: constant series")
        name = p.lower()
        cols[name] = pd.Series(v[:, 1, 1], index=t)
        cols[name + "_3x3"] = pd.Series(v.mean(axis=(1, 2)), index=t)
    df = pd.DataFrame(cols)
    if df.index.has_duplicates or len(df) != 120:
        raise Invalid(f"{len(df)} hourly steps, expected 120")
    if ((df["pm10"] < 0.98 * df["pm25"] - 0.5).mean() > 0.01):
        raise Invalid("PM10 below PM2.5")
    df.index = df.index + UTC_OFFSET  # UTC -> Tashkent time; a value is the mean of the hour ending at its label
    df.index.name = "time"
    df.attrs = {"run": run, "model": files[want[0]][2], "uploaded_at": max(f[3] for f in files.values()),
                "grid_point": [la, lo], "files": sorted(f[0] for f in files.values()),
                "fetched_at": pd.Timestamp.now(tz="Asia/Tashkent").isoformat()}
    return df


def fetch() -> int:
    """Copy every run in the bucket that we do not have yet. Returns the number of runs skipped."""
    DIR.mkdir(parents=True, exist_ok=True)
    s = _session()
    have = {p.stem for p in DIR.glob("????????.parquet")}
    todo = [r for r in runs_available(s) if r not in have]
    skipped = 0
    for run in todo:
        t0 = time.time()
        try:
            df = fetch_run(s, run)
        except Invalid as e:
            print(f"WARNING: SILAM run {run} not saved: {e}", flush=True)
            skipped += 1
            continue
        except Exception as e:  # noqa: BLE001 - network or format trouble: try again tomorrow
            print(f"WARNING: SILAM run {run} not saved: {type(e).__name__}: {e}", flush=True)
            skipped += 1
            continue
        path = DIR / f"{run}.parquet"
        if path.exists():  # written once, never replaced
            continue
        to_parquet(df, path)
        print(f"SILAM {run} ({df.attrs['model']}): saved, PM2.5 {df['pm25'].min():.0f}–{df['pm25'].max():.0f} µg/m³ "
              f"({time.time() - t0:.0f}s)", flush=True)
    if not todo:
        print("SILAM: nothing new", flush=True)
    return skipped


# ---------------------------------------------------------------- scoring
def load() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in sorted(DIR.glob("????????.parquet"))}


def _daily(s: pd.Series) -> pd.Series:
    """Daily mean of hourly AQI over the 24 labels 00..23 (as our daily values and tashkentairdata.uz)."""
    g = s.groupby(s.index.normalize())
    return g.mean()[g.count() == 24]


CORR_DAYS, CORR_MIN_HOURS = 7, 72


def corrected(runs: dict[str, pd.DataFrame], pm: pd.Series) -> dict[str, pd.Series]:
    """Hourly AQI of each run scaled by measured / SILAM over the 7 days before 06:00 of its run
    day; a run with fewer than 72 measured hours in that window gets none."""
    h = pd.Timedelta(hours=1)
    # SILAM's own view of each past hour: the run of that day, hours 06:00 .. 05:00 next day
    own = pd.concat([df["pm25"][(df.index >= pd.Timestamp(r) + 6 * h) & (df.index < pd.Timestamp(r) + 30 * h)]
                     for r, df in sorted(runs.items())])
    both = pd.DataFrame({"act": pm, "s": own[~own.index.duplicated(keep="last")]}).dropna()
    out = {}
    for r, df in runs.items():
        issue = pd.Timestamp(r) + 6 * h
        w = both[(both.index > issue - pd.Timedelta(days=CORR_DAYS)) & (both.index <= issue)]
        if len(w) >= CORR_MIN_HOURS:
            k = w["act"].mean() / w["s"].mean()
            out[r] = pd.Series(aqi.pm_to_aqi(df["pm25"].values * k), index=df.index)
    return out


def _stats(df: pd.DataFrame, cols=("ours", "silam", "corr")) -> dict:
    out = {"n": int(len(df))}
    for c in cols:
        e = df[c] - df["act"]
        out[c] = round(float(e.abs().mean()), 1)
        out[c + "_bias"] = round(float(e.mean()), 1)
        if len(df) >= 5:
            out[c + "_r"] = round(float(np.corrcoef(df[c], df["act"])[0, 1]), 2)
    return out


def score() -> dict:
    """Head to head on our issued forecasts, plus SILAM's own record over every saved run."""
    from .data import load_target
    from .live import _by_day, issue_day
    from .models import daily_aqi

    runs = load()
    if not runs:
        return {}
    pm = load_target()[0]["pm"]
    obs = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    day_act = daily_aqi(pm)
    s_aqi = {r: pd.Series(aqi.pm_to_aqi(df["pm25"].values), index=df.index) for r, df in runs.items()}
    uploaded = {r: pd.Timestamp(df.attrs["uploaded_at"]) for r, df in runs.items()}
    s_corr = corrected(runs, pm)
    nan = pd.Series(dtype=float)

    hrows, drows, n = [], [], 0
    for _, js in _by_day():
        # SILAM's run of the same day; a day we missed is skipped, never paired with an older run
        r = f"{issue_day(js):%Y%m%d}"
        if r not in runs:
            continue
        n += 1
        o, sa, sc = pd.Timestamp(js["origin"]), s_aqi[r], s_corr.get(r, nan)
        for row in js["hourly_7d"]:
            t = pd.Timestamp(row["t"])
            if t in sa.index:
                hrows.append({"h": int((t - o) / pd.Timedelta(hours=1)), "ours": row["aqi"], "silam": sa[t],
                              "corr": sc.get(t, np.nan), "act": obs.get(t, np.nan)})
        sd, scd, today = _daily(sa), _daily(sc) if len(sc) else nan, issue_day(js)
        for row in js["daily_90d"]:
            t = pd.Timestamp(row["t"]).normalize()
            lead = (t - today).days
            if lead in (1, 2, 3) and t in sd.index:
                drows.append({"lead": lead, "date": t, "ours": row["aqi"], "silam": sd[t], "corr": scd.get(t, np.nan),
                              "act": day_act.get(t, np.nan)})
    out = {"runs": len(runs), "first_run": min(runs), "last_run": max(runs), "model": runs[max(runs)].attrs.get("model"),
           "n_forecasts": n}
    h = pd.DataFrame(hrows).dropna()
    for lo, hi, name in ((1, 24, "h1_24"), (25, 72, "h25_72")):
        g = h[(h["h"] >= lo) & (h["h"] <= hi)] if len(h) else h
        if len(g) >= 24:
            out[name] = _stats(g)
    d = pd.DataFrame(drows).dropna()
    for lead in (1, 2, 3):
        g = d[d["lead"] == lead] if len(d) else d
        if len(g):
            out[f"lead{lead}"] = _stats(g)
    # SILAM alone, as is and corrected: every saved run that has a correction, the next Tashkent
    # day after its upload, against the fact
    pub = {r: uploaded[r].tz_convert("Asia/Tashkent").tz_localize(None).normalize() for r in runs}
    rows = []
    for r, sc in s_corr.items():
        day = pub[r] + pd.Timedelta(days=1)
        sd, scd = _daily(s_aqi[r]), _daily(sc)
        if day in sd.index and not np.isnan(day_act.get(day, np.nan)):
            rows.append({"date": day, "silam": sd[day], "corr": scd[day], "act": day_act[day]})
    if rows:
        a = pd.DataFrame(rows)
        out["alone"] = _stats(a, cols=("silam", "corr")) | {"from": f"{a['date'].min():%Y-%m-%d}", "to": f"{a['date'].max():%Y-%m-%d}"}
    # by day, for the chart: the fact (complete days only) and, of each side, the newest forecast
    # published before the day began — for past days the forecast for tomorrow, then the latest
    ours, sil, cor = {}, {}, {}
    for _, js in _by_day():  # oldest first, so a newer forecast overwrites an older one
        d0 = issue_day(js)
        for row in js["daily_90d"]:
            t = pd.Timestamp(row["t"]).normalize()
            if t > d0 and row["kind"] == "model":  # days 8+ are climatology, not a forecast
                ours[t] = (row["aqi"], d0)
    for r in sorted(runs):
        for src, series in ((sil, s_aqi[r]), (cor, s_corr.get(r, nan))):
            for t, v in (_daily(series).items() if len(series) else ()):
                if t > pub[r]:
                    src[t] = (round(float(v)), pub[r])
    if sil:
        days = []
        for t in pd.date_range(min(sil), max(ours.keys() | sil.keys())):
            a = day_act.get(t, np.nan)
            row = {"date": f"{t:%Y-%m-%d}", "act": None if np.isnan(a) else round(float(a))}
            for k, src in (("ours", ours), ("silam", sil), ("corr", cor)):
                v, d0 = src.get(t, (None, None))
                row[k], row[k + "_from"] = v, d0 and f"{d0:%Y-%m-%d}"
            days.append(row)
        out["days"] = days
    write_text(SCORE, json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "score"
    if what == "fetch":
        fetch()
    elif what == "score":
        print(json.dumps(score(), ensure_ascii=False, indent=1))
    else:
        raise SystemExit("usage: python -m aqf.silam fetch|score")
