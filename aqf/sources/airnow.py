"""US Embassy Tashkent PM2.5 monitor (site UZB010001) from the public AirNow file archive.

AirNow publishes one file per UTC hour with every monitor in the network, sorted by
site id. UZB010001 sorts at the very end, so an HTTP range request for the last few KB
of each file is enough — no need to download the full ~700 KB file.

Coverage: ~2018 .. early 2025 (the State Department feed stopped in spring 2025).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://files.airnowtech.org/airnow"
SITE = "UZB010001"
TAIL_BYTES = 4000

_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=64))


def _fetch_tail(url: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r = _session.get(url, headers={"Range": f"bytes=-{TAIL_BYTES}"}, timeout=30)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content.decode("latin-1")
        except requests.RequestException:
            time.sleep(1 + 2 * attempt)
    return None


def _hour_url(ts: pd.Timestamp) -> str:
    return f"{BASE}/{ts:%Y}/{ts:%Y%m%d}/HourlyData_{ts:%Y%m%d%H}.dat"


def _parse_hourly(ts_utc: pd.Timestamp, text: str | None) -> dict | None:
    if not text:
        return None
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) >= 8 and parts[2] == SITE and parts[5] == "PM2.5":
            try:
                return {"time_utc": ts_utc, "pm25": float(parts[7])}
            except ValueError:
                return None
    return None


def download_hourly(start: str, end: str, out: Path, months: set[int] | None = None,
                    workers: int = 24) -> pd.DataFrame:
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    if months:
        hours = hours[hours.month.isin(sorted(months))]
    done: dict[pd.Timestamp, float] = {}
    if out.exists():
        prev = pd.read_parquet(out)
        done = dict(zip(prev["time_utc"], prev["pm25"]))
        # hours fetched before without a value are stored as NaN and not refetched
        hours = hours[~hours.isin(list(done.keys()))]
    print(f"fetching {len(hours)} hourly files ...", flush=True)
    rows = [{"time_utc": k, "pm25": v} for k, v in done.items()]
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_fetch_tail, _hour_url(ts)): ts for ts in hours}
        for i, f in enumerate(cf.as_completed(futs), 1):
            ts = futs[f]
            rec = _parse_hourly(ts, f.result())
            rows.append(rec or {"time_utc": ts, "pm25": float("nan")})
            if i % 2000 == 0:
                print(f"  {i}/{len(hours)}", flush=True)
                _save(rows, out)
    return _save(rows, out)


def _save(rows: list[dict], out: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows).drop_duplicates("time_utc").sort_values("time_utc")
    df.to_parquet(out, index=False)
    return df


def load_hourly(path: Path) -> pd.Series:
    """Hourly PM2.5 (µg/m³) indexed by Tashkent local time (UTC+5, no DST).

    AirNow labels an hourly average by the hour it starts; the IQAir/tashkentairdata series
    labels it by the hour it ends (cross-correlation peaks at +1 h), so shift by one hour.
    """
    df = pd.read_parquet(path)
    s = df.set_index("time_utc")["pm25"]
    s.index = s.index.tz_convert("Asia/Tashkent").tz_localize(None) + pd.Timedelta(hours=1)
    # AirNow uses -999 and similar sentinels for missing/invalid values
    s = s.where((s >= 0) & (s < 1000))
    return s.sort_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-09-01")
    ap.add_argument("--end", default="2025-05-01")
    ap.add_argument("--months", default="9,10,11,12,1,2,3,4")
    ap.add_argument("--out", default="data/raw/airnow/tashkent_embassy_pm25.parquet")
    a = ap.parse_args()
    months = {int(m) for m in a.months.split(",")} if a.months else None
    df = download_hourly(a.start, a.end, Path(a.out), months)
    print(df["pm25"].notna().sum(), "valid hours of", len(df))
