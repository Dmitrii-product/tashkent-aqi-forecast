"""Export the served LightGBM models for publication (owner, 2026-11-06).

Writes a self-contained English copy of the models in data/processed/models.pkl to
output/model_export/<name>/, in formats that are safe to load and keep no measurement data:

  lightgbm/h<lo>-<hi>.txt  LightGBM's own text format, one model per horizon group (exact)
  norms.json              the multi-year norms the models read: the hourly norm (365 x 24,
                          log1p PM2.5) and its daily mean; for days 8-90 the daily norm (log AQI),
                          the monthly spread quantiles and the red/purple-day frequency by date
  intervals.json          the 80% bands the site uses
  config.json             horizon routing, feature lists, parameters, versions, verification
  README.md, LICENSE.md   model card and licence (CC BY-NC 4.0), in English
  SHA256SUMS

Left out on purpose: the per-season sums the norms keep for leave-one-season-out training (the
hourly measurement history could be rebuilt from them), the CAMS series one model carries
(Copernicus data), and the neural network blended in since model version 2026-11-04 (retrained
every morning, never stored).

models.pkl is only read, and the site's forecast never uses the export. Every export is written
to <name>.partial and gets its final name only after all checks pass (nothing is deleted when one
fails):
  verify      the models rebuilt from the exported files alone forecast exactly what models.pkl
              forecasts from the inputs of the latest issued forecast;
  leak_check  (owner: "tests that guarantee other people's data do not leak, run at every export")
              1. only the expected files, each with a fixed structure: norms of fixed shapes (no
                 per-season axis), short number lists in config/intervals, no pickle, LightGBM
                 files with the configured features and no embedded categorical data;
              2. no run of 8 consecutive values (12 at whole numbers, 10 at one decimal; all but two
                 distinct, not an arithmetic progression) of a source series appears among the
                 consecutive numbers of any exported file (JSON arrays row- and column-wise, tree
                 thresholds and leaf values, numbers in the texts, also every 2nd-4th = table columns), at 0-4 decimals: measured PM2.5,
                 log1p PM2.5 and AQI (hourly and daily), CAMS, the weather archive, airport wind;
              3. no local paths, e-mail addresses or tokens;
  self_test   the leak check must catch planted leaks in a copy of the export (measurements in a
              norm table, a per-season table, daily AQI in the card, measurements in tree leaves, CAMS
              in the config, a pickle file); if it misses one, the export fails.

    .venv/bin/python -m aqf.export                 # export + verify + leak check + self test
    .venv/bin/python -m aqf.export --check <dir>   # leak check and checksums of an existing export
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import shutil
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import aqi
from .data import PROC, RAW, ROOT, load_target, load_weather
from .features import Climatology, weather_features
from .models import (H_GROUPS, HOURLY_VARIANTS, MODEL_VERSIONS, HorizonRouter, HourlyModel,
                     LongRange, SeasonalMean, daily_aqi)

OUT = ROOT / "output" / "model_export"
LOG = ROOT / "data" / "forecast_log"
MODELS = PROC / "models.pkl"
INTERVALS = PROC / "intervals.json"
SUMMARY = PROC / "backtest_summary.json"
AUTHOR = "ProductDmitrii"  # the owner's Hugging Face account; the card links to no website (owner, 2026-11-06)


def _floats(a) -> list:
    a = np.asarray(a, float)
    assert np.isfinite(a).all(), "norm tables must be complete"
    return a.tolist()  # Python floats: json writes them so that they read back exactly


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------- export
def export(dest: Path | None = None) -> Path:
    m = pickle.loads(MODELS.read_bytes())
    router, lr = m["hourly"], m["longrange"]
    name = f"tashkent-aqi-lightgbm-v{m['version']}-trained-{str(m['trained_at'])[:10]}"
    dest = Path(dest) if dest else OUT / name
    if dest.exists():
        raise FileExistsError(f"{dest} exists: export to a new folder (nothing is overwritten)")
    if dest.suffix != ".partial":
        dest = dest.with_name(dest.name + ".partial")
        if dest.exists():
            raise FileExistsError(f"{dest} exists: a failed export is kept for inspection; remove it first")
    (dest / "lightgbm").mkdir(parents=True)

    groups, tables = {}, []
    for (lo, hi), v in sorted(router.choice.items()):
        hm = router.variants[v]
        key = f"{lo}-{hi}"
        b = hm.boosters[(lo, hi)]
        b = getattr(b, "booster_", b)  # the scikit-learn wrapper holds a plain Booster
        feats = hm.features if isinstance(hm.features, list) else hm.features[(lo, hi)]
        assert b.feature_name() == [f"Column_{i}" for i in range(len(feats))] or b.feature_name() == list(feats)
        text = re.sub(r"\bColumn_(\d+)\b", lambda mm: feats[int(mm.group(1))], b.model_to_string())
        (dest / "lightgbm" / f"h{key}.txt").write_text(text)
        groups[key] = {"hours": [lo, hi], "file": f"lightgbm/h{key}.txt", "variant": v,
                       "predicts": f"log1p(PM2.5 at t+h) - {hm.base}", "base": hm.base,
                       "uses_cams": getattr(hm, "cams", None) is not None,
                       "trees": b.num_trees(), "features": list(feats),
                       "training": {k: val for k, val in HOURLY_VARIANTS[v].items() if k != "use_cams"}}
        tables.append(hm.clim)
    assert [tuple(g["hours"]) for g in groups.values()] == list(H_GROUPS), "one model per horizon group"
    c0 = tables[0]
    assert all(np.array_equal(c.table, c0.table) and np.array_equal(c.daily, c0.daily) for c in tables), \
        "all hourly models read the same norm"

    norms = {
        "hourly_norm_log1p_pm25": {"shape": [365, 24], "index": "day of year (1..365, Feb 29 -> Mar 1 row) x hour 0..23",
                                   "smoothing_days": c0.win, "values": _floats(c0.table)},
        "hourly_norm_daily_mean": {"shape": [365], "values": _floats(c0.daily)},
        "longrange_norm_log_aqi": {"shape": [365], "smoothing_days": lr.clim.win, "values": _floats(lr.clim.table)},
        "longrange_spread_q10_q90_by_month": {"months": [int(x) for x in lr.q.index],
                                              "q10": _floats(lr.q[0.1].values), "q90": _floats(lr.q[0.9].values)},
        "longrange_quantiles_by_month": {"levels": _floats(np.linspace(0.01, 0.99, 99)),
                                         "values": {str(k): _floats(v) for k, v in sorted(lr.rq.items())}},
        "red_day_frequency_aqi_over_150": {"shape": [365], "values": _floats(lr.freq[150].table)},
        "purple_day_frequency_aqi_over_200": {"shape": [365], "values": _floats(lr.freq[200].table)},
    }
    (dest / "norms.json").write_text(json.dumps(norms))
    shutil.copyfile(INTERVALS, dest / "intervals.json")

    cfg = {
        "name": name, "model_version": m["version"], "trained_at": str(m["trained_at"]),
        "training_data_end": str(m["data_end"]), "lightgbm_version": lgb.__version__,
        "target": "hourly city-mean PM2.5 (ug/m3) in Tashkent, modelled as log1p(PM2.5); "
                  "US AQI from the EPA 2024 PM2.5 breakpoints (continuous piecewise-linear)",
        "hourly": {"horizon_hours": [1, 168], "groups": groups},
        "longrange": {"days": lr.max_d, "anomaly_tau_days": lr.tau, "anomaly_k": lr.k,
                      "product": "days 1-7: daily means of the hourly forecast; days 8-90: "
                                 "exp(norm + day-7 anomaly * exp(-(d - 7) / 4))"},
        "versions": MODEL_VERSIONS,
    }
    (dest / "config.json").write_text(json.dumps(cfg, indent=1))
    (dest / "README.md").write_text(_card(cfg))
    (dest / "LICENSE.md").write_text(_licence())
    return dest


# ---------------------------------------------------------------------------- reload
def load_public(dest: Path) -> dict:
    """The models rebuilt from the exported files only (no models.pkl, no per-season sums)."""
    dest = Path(dest)
    cfg = json.loads((dest / "config.json").read_text())
    norms = json.loads((dest / "norms.json").read_text())
    clim = object.__new__(Climatology)
    clim.win = norms["hourly_norm_log1p_pm25"]["smoothing_days"]
    clim.table = np.array(norms["hourly_norm_log1p_pm25"]["values"])
    clim.daily = np.array(norms["hourly_norm_daily_mean"]["values"])
    variants, choice = {}, {}
    for g in cfg["hourly"]["groups"].values():
        lo, hi = g["hours"]
        hm = variants.get(g["variant"])
        if hm is None:
            hm = object.__new__(HourlyModel)
            hm.max_h, hm.base, hm.clim, hm.boosters, hm.features = 168, g["base"], clim, {}, {}
            hm.cams = pd.DataFrame(columns=["pm2_5", "pm10", "dust"]) if g["uses_cams"] else None  # set_cams fills it
            variants[g["variant"]] = hm
        hm.boosters[(lo, hi)] = lgb.Booster(model_file=str(dest / g["file"]))
        hm.features[(lo, hi)] = g["features"]
        choice[(lo, hi)] = g["variant"]
    sm = lambda vals: _seasonal(np.array(vals))  # noqa: E731
    lr = object.__new__(LongRange)
    lr.max_d, lr.tau, lr.k = cfg["longrange"]["days"], cfg["longrange"]["anomaly_tau_days"], cfg["longrange"]["anomaly_k"]
    lr.clim = sm(norms["longrange_norm_log_aqi"]["values"])
    sp = norms["longrange_spread_q10_q90_by_month"]
    lr.q = pd.DataFrame({0.1: sp["q10"], 0.9: sp["q90"]}, index=sp["months"])
    lr.rq = {int(k): np.array(v) for k, v in norms["longrange_quantiles_by_month"]["values"].items()}
    lr.freq = {150: sm(norms["red_day_frequency_aqi_over_150"]["values"]),
               200: sm(norms["purple_day_frequency_aqi_over_200"]["values"])}
    return {"hourly": HorizonRouter(variants, choice), "longrange": lr}


def _seasonal(table: np.ndarray) -> SeasonalMean:
    s = object.__new__(SeasonalMean)
    s.table = table
    return s


# ---------------------------------------------------------------------------- verify
def verify(dest: Path) -> dict:
    """Forecast from the latest issued forecast's saved inputs with models.pkl and with the export."""
    dest = Path(dest)
    internal, public = pickle.loads(MODELS.read_bytes()), load_public(dest)
    stem = sorted(LOG.glob("??????????.json"))[-1].with_suffix("")
    origin = pd.Timestamp(json.loads(stem.with_suffix(".json").read_text())["origin"])
    wf = weather_features(pd.read_parquet(f"{stem}_wx.parquet"))
    cams = pd.read_parquet(f"{stem}_cams.parquet")
    pm = load_target()[0]["pm"]
    pm = pm[pm.index <= origin]
    h = []
    for mod in (internal, public):
        mod["hourly"].set_cams(cams)
        h.append(mod["hourly"].predict_frame(pd.DatetimeIndex([origin]), pm, wf))
    day0 = pd.DatetimeIndex([origin.normalize() - pd.Timedelta(days=1)])
    d = [mod["longrange"].predict_frame(day0, pm) for mod in (internal, public)]
    T = pd.DatetimeIndex(d[0].index.get_level_values("origin") + pd.to_timedelta(d[0].index.get_level_values("d"), unit="D"))
    p = [mod["longrange"].prob_above(d[0]["pred_aqi"].values, T.month.values, 150) for mod in (internal, public)]
    f = [mod["longrange"].climate_prob(T, 150) for mod in (internal, public)]
    norms = json.loads((dest / "norms.json").read_text())
    res = {
        "issued_forecast": stem.name, "origin": str(origin),
        "hourly_max_abs_diff_log1p_pm25": float(np.nanmax(np.abs(h[0]["pred_lp"].values - h[1]["pred_lp"].values))),
        "hourly_max_abs_diff_aqi": float(np.nanmax(np.abs(h[0]["pred_aqi"].values - h[1]["pred_aqi"].values))),
        "hourly_hours_compared": int(np.isfinite(h[0]["pred_lp"].values).sum()),
        "longrange_max_abs_diff_aqi": float(np.nanmax(np.abs(d[0][["pred_aqi", "clim_lo", "clim_hi"]].values
                                                           - d[1][["pred_aqi", "clim_lo", "clim_hi"]].values))),
        "red_chance_max_abs_diff": float(max(np.nanmax(np.abs(p[0] - p[1])), np.nanmax(np.abs(f[0] - f[1])))),
        "largest_table": max(int(np.prod(v["shape"])) for v in norms.values() if "shape" in v),
        "files": {str(q.relative_to(dest)): q.stat().st_size for q in sorted(dest.rglob("*")) if q.is_file()},
    }
    res["identical"] = res["hourly_max_abs_diff_log1p_pm25"] == 0 and res["longrange_max_abs_diff_aqi"] == 0 \
        and res["red_chance_max_abs_diff"] == 0
    cfg_path = dest / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["verification"] = {k: v for k, v in res.items() if k != "files"}
    cfg_path.write_text(json.dumps(cfg, indent=1))
    if not res["identical"]:
        raise RuntimeError(f"exported models do not forecast what models.pkl does: {cfg['verification']}")
    return res


def _write_sums(dest: Path) -> None:
    sums = [f"{_sha(q)}  {q.relative_to(dest).as_posix()}" for q in sorted(dest.rglob("*"))
            if q.is_file() and q.name != "SHA256SUMS"]
    (dest / "SHA256SUMS").write_text("\n".join(sums) + "\n")


def _check_sums(dest: Path) -> None:
    listed = {}
    for line in (dest / "SHA256SUMS").read_text().splitlines():
        h, f = line.split("  ", 1)
        listed[f] = h
    present = {q.relative_to(dest).as_posix() for q in dest.rglob("*") if q.is_file() and q.name != "SHA256SUMS"}
    if set(listed) != present:
        raise LeakError([f"files not matching SHA256SUMS: {sorted(set(listed) ^ present)}"])
    bad = [f for f, h in listed.items() if _sha(dest / f) != h]
    if bad:
        raise LeakError([f"checksum mismatch: {bad}"])


# ---------------------------------------------------------------------------- leak check
class LeakError(RuntimeError):
    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("export must not be published:\n  " + "\n  ".join(problems[:20]))


DPS = (0, 1, 2, 3, 4)
KS = {0: 12, 1: 10, 2: 8, 3: 8, 4: 8}  # window length by number of decimals: coarse rounding needs longer runs
P = np.uint64(1_000_003)
FILES = re.compile(r"^(README\.md|LICENSE\.md|config\.json|norms\.json|intervals\.json|SHA256SUMS|lightgbm/h\d+-\d+\.txt)$")
NORMS = {  # key -> {field: shape}; nothing else may be in norms.json
    "hourly_norm_log1p_pm25": {"values": (365, 24)}, "hourly_norm_daily_mean": {"values": (365,)},
    "longrange_norm_log_aqi": {"values": (365,)},
    "longrange_spread_q10_q90_by_month": {"months": (12,), "q10": (12,), "q90": (12,)},
    "longrange_quantiles_by_month": {"levels": (99,), **{f"values.{m}": (99,) for m in range(1, 13)}},
    "red_day_frequency_aqi_over_150": {"values": (365,)}, "purple_day_frequency_aqi_over_200": {"values": (365,)},
}
NORM_META = {"shape", "index", "smoothing_days"}
CONFIG_KEYS = {"name", "model_version", "trained_at", "training_data_end", "lightgbm_version", "target", "hourly",
               "longrange", "versions", "verification", "leak_check"}
GROUP_KEYS = {"hours", "file", "variant", "predicts", "base", "uses_cams", "trees", "features", "training"}
SECRETS = re.compile(r"/(?:Users|home)/|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}|\b(?:sk|ghp|hf|vercel)_[A-Za-z0-9]{16,}"
                     r"|\.env\b|BEGIN [A-Z ]*PRIVATE KEY")
NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
TREE_ARRAYS = ("threshold=", "leaf_value=", "internal_value=", "leaf_weight=", "internal_weight=", "split_gain=")


def _hashes(x, dp: int) -> np.ndarray:
    """64-bit hashes of every window of KS[dp] consecutive values rounded to `dp` decimals that has
    at least KS[dp] - 2 different values and is not an arithmetic progression (flat stretches,
    runs of zeros and counters like 1, 2, 3 say nothing)."""
    k = KS[dp]
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < k:
        return np.empty(0, np.uint64)
    W = sliding_window_view(np.round(np.clip(x, -1e12, 1e12) * 10.0 ** dp).astype(np.int64), k)
    d = np.diff(W, axis=1)
    keep = ((np.diff(np.sort(W, axis=1), axis=1) != 0).sum(axis=1) + 1 >= k - 2) & ~(d == d[:, :1]).all(axis=1)
    W = np.ascontiguousarray(W[keep]).view(np.uint64)
    with np.errstate(over="ignore"):
        return (W * (P ** np.arange(k, dtype=np.uint64))).sum(axis=1)


@lru_cache(maxsize=1)
def _sources() -> dict[int, np.ndarray]:
    """Hashes of the third-party series the models were made from, per number of decimals."""
    pm = load_target()[0]["pm"].dropna()
    a = pd.Series(aqi.pm_to_aqi(pm.values), index=pm.index)
    da = daily_aqi(load_target()[0]["pm"]).dropna()
    series = [(pm, (0, 1, 2)), (np.log1p(pm), (2, 3, 4)), (a, (0, 1, 2)), (da, (0, 1, 2)), (np.log(da), (2, 3, 4))]
    cams = pd.read_parquet(RAW / "weather" / "cams.parquet")
    series += [(cams[c], (1, 2)) for c in cams.columns] + [(np.log1p(cams[c]), (3, 4)) for c in cams.columns]
    w = load_weather(upper=False)
    series += [(w[c].astype(float), (1, 2)) for c in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m",
                                                       "precipitation", "surface_pressure", "wind_gusts_10m")]
    metar = RAW / "weather" / "metar_UZTT_full.csv"
    if metar.exists():
        m = pd.read_csv(metar, na_values="M", usecols=["sknt", "tmpc"])
        series += [((m["sknt"] * 1.852).values, (1, 2)), (m["tmpc"].values, (1,))]
    out = {dp: [] for dp in DPS}
    for x, dps in series:
        for dp in dps:
            out[dp].append(_hashes(np.asarray(x, float), dp))
    return {dp: np.unique(np.concatenate(v)) for dp, v in out.items() if v}


def _json_sequences(o, path=""):
    """Every list of numbers in a JSON document; 2-D tables also column by column."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _json_sequences(v, f"{path}.{k}")
    elif isinstance(o, list) and o:
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in o):
            yield path, np.array(o, float)
        elif all(isinstance(x, list) for x in o) and all(all(isinstance(y, (int, float)) for y in x) for x in o) \
                and len({len(x) for x in o}) == 1:
            t = np.array(o, float)
            yield path + "[rows]", t.ravel()
            yield path + "[cols]", t.T.ravel()
        else:
            for i, v in enumerate(o):
                yield from _json_sequences(v, f"{path}[{i}]")


def _shape(o) -> tuple:
    return np.array(o, float).shape


def leak_check(dest: Path) -> dict:
    """Raise LeakError if the export holds anything but the models and multi-year aggregates."""
    dest, problems = Path(dest), []
    files = sorted(q.relative_to(dest).as_posix() for q in dest.rglob("*") if q.is_file())
    problems += [f"unexpected file: {f}" for f in files if not FILES.match(f)]
    cfg = json.loads((dest / "config.json").read_text())
    expected_models = {g["file"] for g in cfg.get("hourly", {}).get("groups", {}).values()}
    present_models = {f for f in files if f.startswith("lightgbm/")}
    if expected_models != present_models or len(present_models) != len(H_GROUPS):
        problems.append(f"model files differ from config: {sorted(expected_models ^ present_models)}")
    for f in files:
        b = (dest / f).read_bytes()
        if b[:2] in (b"\x80\x03", b"\x80\x04", b"\x80\x05") or b[:4] == b"PK\x03\x04" or b"\x00" in b[:4096]:
            problems.append(f"binary or pickled content: {f}")
        elif SECRETS.search(b.decode("utf-8", "replace")):
            problems.append(f"path, e-mail or token in {f}: {SECRETS.search(b.decode('utf-8', 'replace')).group(0)!r}")

    # 1. structure
    norms = json.loads((dest / "norms.json").read_text())
    if set(norms) != set(NORMS):
        problems.append(f"norms.json keys differ: {sorted(set(norms) ^ set(NORMS))}")
    for key, fields in NORMS.items():
        item = norms.get(key, {})
        flat = {k: v for k, v in item.items() if k not in NORM_META and k != "values" or (k == "values" and not isinstance(v, dict))}
        if isinstance(item.get("values"), dict):
            flat.update({f"values.{m}": v for m, v in item["values"].items()})
        if set(flat) != set(fields):
            problems.append(f"norms.json[{key}] fields differ: {sorted(set(flat) ^ set(fields))}")
        for fld, shp in fields.items():
            if fld in flat and _shape(flat[fld]) != shp:
                problems.append(f"norms.json[{key}].{fld} has shape {_shape(flat[fld])}, expected {shp}")
    if set(cfg) - CONFIG_KEYS:
        problems.append(f"config.json has unexpected keys: {sorted(set(cfg) - CONFIG_KEYS)}")
    for k, g in cfg.get("hourly", {}).get("groups", {}).items():
        if set(g) - GROUP_KEYS:
            problems.append(f"config.json group {k} has unexpected keys: {sorted(set(g) - GROUP_KEYS)}")
    for path, x in _json_sequences(cfg):
        if len(x) > 12:
            problems.append(f"config.json{path}: a list of {len(x)} numbers")
    iv = json.loads((dest / "intervals.json").read_text())
    if set(iv) - {"hourly", "daily", "daily_q"}:
        problems.append(f"intervals.json has unexpected keys: {sorted(set(iv) - {'hourly', 'daily', 'daily_q'})}")
    for part in iv.values():
        if not isinstance(part, list) or len(part) > 30:
            problems.append("intervals.json: unexpected structure")
            continue
        for e in part:
            if set(e) - {"lo", "hi", "q10", "q90", "n", "q", "live_adjusted"} or len(e.get("q", [])) > 99 \
                    or not re.fullmatch(r"(\d{4}-\d{2}-\d{2})?", str(e.get("live_adjusted", ""))):
                problems.append(f"intervals.json entry with unexpected content: {sorted(e)[:6]}")
    for g in cfg.get("hourly", {}).get("groups", {}).values():
        f = dest / g["file"]
        if not f.exists():
            continue
        text = f.read_text()
        names = re.search(r"^feature_names=(.*)$", text, re.M)
        if not names or names.group(1).split(" ") != g["features"]:
            problems.append(f"{g['file']}: feature names differ from config.json")
        cat = re.search(r"^pandas_categorical:(.*)$", text, re.M)
        if cat and cat.group(1).strip() not in ("null", "[]"):
            problems.append(f"{g['file']}: embedded categorical data")

    # 2. no run of a source series anywhere
    seqs = []
    for f in ("norms.json", "intervals.json", "config.json"):
        seqs += [(f + p, x) for p, x in _json_sequences(json.loads((dest / f).read_text()))]
    for f in ("README.md", "LICENSE.md"):  # numbers in reading order, and every 2nd/3rd/4th (table columns)
        x = np.array([float(t) for t in NUM.findall((dest / f).read_text())], float)
        seqs += [(f"{f}[every {st} from {j}]", x[j::st]) for st in (1, 2, 3, 4) for j in range(st)]
    for g in cfg.get("hourly", {}).get("groups", {}).values():
        f = dest / g["file"]
        if f.exists():  # each kind of tree array, all trees of the file one after another
            arrays = {k: [] for k in TREE_ARRAYS}
            for line in f.read_text().splitlines():
                if line.startswith(TREE_ARRAYS):
                    key, vals = line.split("=", 1)
                    arrays[key + "="].append(np.array(vals.split(" "), float))
            seqs += [(f"{g['file']}:{k[:-1]}", np.concatenate(v)) for k, v in arrays.items() if v]
    src = _sources()
    windows = 0
    for dp in DPS:  # all windows of all sequences at once, one lookup per number of decimals
        hs = [_hashes(x, dp) for _, x in seqs]
        windows += sum(len(h) for h in hs)
        if dp not in src or not windows:
            continue
        owner = np.repeat(np.arange(len(seqs)), [len(h) for h in hs])
        hit = np.isin(np.concatenate(hs), src[dp])
        for i in np.unique(owner[hit]):
            problems.append(f"{seqs[i][0]}: {int(hit[owner == i].sum())} run(s) of {KS[dp]} values of a source series "
                            f"(rounded to {dp} decimals)")
    if problems:
        raise LeakError(problems)
    return {"files": len(files), "sequences": len(seqs), "windows_checked": windows,
            "source_windows": int(sum(len(v) for v in src.values()))}


def self_test(dest: Path) -> list[str]:
    """Plant leaks in a copy of the export; the leak check must catch every one."""
    pm = load_target()[0]["pm"].dropna()
    pm = pm[pm.index >= "2025-11-01"]
    da = daily_aqi(load_target()[0]["pm"]).dropna()
    cams = pd.read_parquet(RAW / "weather" / "cams.parquet")["pm2_5"].dropna()

    def edit_json(f, fn):
        return lambda d: (d / f).write_text(json.dumps(fn(json.loads((d / f).read_text()))))

    def norm_with_measurements(n):
        v = np.array(n["hourly_norm_log1p_pm25"]["values"])
        v.ravel()[100:148] = np.log1p(pm.values[:48])
        n["hourly_norm_log1p_pm25"]["values"] = v.tolist()
        return n

    def leaves_with_measurements(d):
        f = d / "lightgbm" / "h1-1.txt"
        lines = f.read_text().splitlines()
        i = next(k for k, l in enumerate(lines) if l.startswith("leaf_value="))
        n = len(lines[i].split("=", 1)[1].split(" "))
        vals = list(np.round(pm.values[200:200 + n], 2)) + [0.0] * max(0, n - 20)
        lines[i] = "leaf_value=" + " ".join(f"{x:g}" for x in vals[:n])
        f.write_text("\n".join(lines) + "\n")

    plants = {
        "hourly measurements inside the norm table": edit_json("norms.json", norm_with_measurements),
        "a per-season norm table": edit_json("norms.json", lambda n: {**n, "hourly_norm_by_season": {
            "values": np.zeros((9, 365, 24)).tolist()}}),
        "daily AQI as a table in the model card": lambda d: (d / "README.md").write_text(
            (d / "README.md").read_text() + "\n| day | AQI |\n|---|---|\n" + "\n".join(
                f"| {i} | {round(v)} |" for i, v in enumerate(da.values[-40:]))),
        "measurements in tree leaf values": leaves_with_measurements,
        "CAMS series in config.json": edit_json("config.json", lambda c: {**c, "hourly": {
            **c["hourly"], "groups": {**c["hourly"]["groups"], "1-1": {
                **c["hourly"]["groups"]["1-1"], "training": {"cams": cams.values[:30].round(2).tolist()}}}}}),
        "a pickled file": lambda d: (d / "models.pkl").write_bytes(pickle.dumps({"pm": pm.values[:10]})),
    }
    caught = []
    with tempfile.TemporaryDirectory() as tmp:
        for what, plant in plants.items():
            d = Path(tmp) / "copy"
            shutil.copytree(dest, d)
            plant(d)
            try:
                leak_check(d)
            except LeakError as e:
                caught.append(f"{what}: caught ({e.problems[0][:90]})")
            else:
                raise RuntimeError(f"the leak check missed a planted leak: {what}")
            shutil.rmtree(d)  # the temporary copy only
    return caught


def run(dest: Path | None = None) -> Path:
    """Export, verify, leak-check and self-test in a .partial folder, then give it its name."""
    staging = export(dest)
    final = staging.with_name(staging.name.removesuffix(".partial"))
    res = verify(staging)
    leak = leak_check(staging)
    controls = self_test(staging)
    cfg_path = staging / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["leak_check"] = {**leak, "self_test": controls}
    cfg_path.write_text(json.dumps(cfg, indent=1))
    leak_check(staging)  # the final config too
    _write_sums(staging)
    staging.rename(final)
    print(json.dumps({"export": str(final), "verification": {k: v for k, v in res.items() if k != "files"},
                      "leak_check": cfg["leak_check"]}, indent=1))
    return final


# ---------------------------------------------------------------------------- texts
def _licence() -> str:
    return f"""# Licence

The model files and tables in this folder (`lightgbm/`, `norms.json`, `intervals.json`,
`config.json`) are © 2026 {AUTHOR} and are licensed under the
**Creative Commons Attribution-NonCommercial 4.0 International** licence (CC BY-NC 4.0):
https://creativecommons.org/licenses/by-nc/4.0/legalcode

You may share and adapt them for non-commercial purposes, giving appropriate credit
("Tashkent air-quality forecast, {AUTHOR}") and indicating changes.

The models were trained on third-party data that are **not** included here and are not covered
by this licence: IQAir measurements (via tashkentairdata.uz; IQAir's terms allow personal,
non-commercial use), the U.S. Embassy Tashkent monitor (AirNow), Open-Meteo weather forecasts and
ERA5 reanalysis (CC BY 4.0), and CAMS forecasts (Copernicus Atmosphere Monitoring Service).

THE MATERIAL IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. See the disclaimer in README.md.
"""


def _card(cfg: dict) -> str:
    s = json.loads(SUMMARY.read_text())
    hl, hourly = s["headline"], s["hourly"]["iqair"]
    rows_h = "\n".join(f"| {r['b']} h | ±{r['model']:.1f} | ±{r['persist']:.1f} | ±{r['clim']:.1f} |" for r in hourly)
    rows_d = "\n".join(f"| {r['b']} | ±{r['model']:.1f} | ±{r['clim']:.1f} |" for r in s["daily"])
    gb = cfg["hourly"]["groups"]
    rows_g = "\n".join(f"| {k} | `{g['file']}` | {g['trees']} | {len(g['features'])} | {'yes' if g['uses_cams'] else 'no'} |"
                       for k, g in gb.items())
    v = next(x for x in MODEL_VERSIONS if x["id"] == cfg["model_version"])
    rows_v = "\n".join(f"| {b['period']} | ±{b['mae']:.2f} | {b['diff']:+.2f} ({b['ci'][0]:+.2f}…{b['ci'][1]:+.2f}) |"
                       for b in v.get("backtest", []))
    feats = sorted({f for g in gb.values() for f in g["features"]})
    return f"""---
license: cc-by-nc-4.0
language: en
library_name: lightgbm
pipeline_tag: tabular-regression
tags: [air-quality, pm2.5, aqi, forecasting, time-series, tashkent, uzbekistan, lightgbm]
---

# Tashkent air-quality forecast: LightGBM models (version {cfg['model_version']})

**A non-commercial research project.** Hourly forecast of the city-mean US AQI (from PM2.5) in
Tashkent, Uzbekistan, 1 to 168 hours ahead, plus a 90-day daily outlook. Snapshot: weekly
retraining of {cfg['trained_at'][:10]}, measurements up to {cfg['training_data_end'][:16]} (Tashkent time).

> **Disclaimer.** Not an official forecast, not a warning service and not medical advice. The
> forecast can be wrong, sometimes badly (sudden dust, smoke or inversions, days ahead). Use
> official sources and your doctor's advice for health decisions. Provided "as is", without any
> warranty; the author is not liable for any use of it.

## What is in this folder

| File | Content |
|---|---|
| `lightgbm/h<lo>-<hi>.txt` | 14 LightGBM models in LightGBM's own text format, one per horizon group |
| `norms.json` | seasonal norms the models read (multi-year averages only) and the 90-day product's tables |
| `intervals.json` | quantiles of past errors used for the 80% ranges and the chance of a red day |
| `config.json` | horizon routing, each model's ordered feature list, training settings, versions, verification |
| `LICENSE.md`, `SHA256SUMS` | licence and checksums |

No measurement data are included. The norms are averages over 2018–2026 by day of year and hour;
the per-season sums kept for training were removed. Loading needs no pickle:
`lightgbm.Booster(model_file="lightgbm/h1-1.txt")`, `json.load(open("norms.json"))`.

## How the forecast is made

* **Target.** City-mean hourly PM2.5 in Tashkent (mean over IQAir sensors, via
  tashkentairdata.uz; before 2025 the U.S. Embassy monitor, rescaled to the IQAir level), as
  `log1p(PM2.5)`. US AQI from the EPA 2024 PM2.5 breakpoints.
* **Hourly, 1–168 h.** One model per horizon group; each predicts the change of `log1p(PM2.5)`
  from the last measured hour (`lag0`): `forecast(t+h) = lag0 + model(features)`.

| Hours | File | Trees | Features | CAMS inputs |
|---|---|---|---|---|
{rows_g}

* **Inputs ({len(feats)} distinct features).** Recent pollution (lags, means, trends, the same hour
  on past days); the seasonal norm (`norms.json`) and the anomaly from it; calendar; the weather
  forecast at the target hour and accumulated since the origin (mean of four Open-Meteo models:
  best match = ECMWF IFS, GFS, ICON, GEM; the archived forecast of the matching lead in
  training), including wind, gusts, westerly flow, calm hours, rain, humidity, cloud, sunshine,
  pressure tendency and inversion strength (850/925/700 hPa minus 2 m temperature); the CAMS
  PM2.5 and dust forecast for 7–24 h. Exact names and order are in `config.json` and in each
  model file; how they are computed is in `aqf/features.py` of the project code (MIT).
* **Days 8–90.** The norm for the date plus the day-7 anomaly decaying with an e-folding time of
  4 days; the chance of a red day from the monthly spread (`norms.json`).
* **Served forecast.** Since model version 2026-11-04 the served forecast blends these models 50/50 (in
  log PM2.5) with a small neural network from hour 7 on. The network is retrained every morning
  and is **not** included.

## Accuracy (walk-forward backtest)

Retrained at every month start on all earlier data, forecasts every 3 hours through each cold
season (October–March), archived weather forecasts as issued from 2024 on. Mean absolute error,
US AQI, winter 2025/26 and October 2026 (IQAir data):

| Horizon | This model | Persistence | Climatology |
|---|---|---|---|
{rows_h}

Daily means, all winters 2020–2026 (before 2024 with actual instead of forecast weather, so
slightly optimistic there):

| Days ahead | This model | Climatology |
|---|---|---|
{rows_d}

Winter 2025/26 headline: ±{hl['h24']['mae']:.1f} AQI for the next 24 hours, ±{hl['h168']['mae']:.1f}
for the week, ±{hl['d17']['mae']:.1f} for daily means of days 1–7. Change against the previous
model version (MAE over 1–168 h, 95% interval over days):

| Period | MAE | Change |
|---|---|---|
{rows_v}

## Limitations

* A city-wide mean, not the value at a particular place.
* Errors of the weather forecast: with a perfect weather forecast (including the wind measured at
  the airport) the error would be about 13% lower 1 day ahead and 19% lower 3 days ahead, and a
  third or more in stagnant spells; the rest is not the weather.
* Sudden episodes (dust storms, fires, abrupt inversions) are seen only through the weather and
  may come late. Level shifts lasting weeks (e.g. autumn 2026) are not anticipated.
* Trained for Tashkent only; not transferable to other cities without retraining.

## Data sources and attribution

* Air quality: IQAir sensors via tashkentairdata.uz (2025–); U.S. Embassy Tashkent monitor via
  AirNow (2018–2025).
* Weather: Open-Meteo (CC BY 4.0) — forecasts of ECMWF IFS, NOAA GFS, DWD ICON, ECCC GEM and the
  ERA5 reanalysis (Copernicus Climate Change Service).
* Atmospheric composition: CAMS, Copernicus Atmosphere Monitoring Service (via Open-Meteo).

None of these data are redistributed here.

## Licence

Model files and tables: CC BY-NC 4.0 (see `LICENSE.md`). Code: MIT.
"""


if __name__ == "__main__":
    if "--check" in sys.argv:
        d = Path(sys.argv[sys.argv.index("--check") + 1])
        _check_sums(d)
        print(json.dumps({"checksums": "ok", "leak_check": leak_check(d)}, indent=1))
    else:
        run()
