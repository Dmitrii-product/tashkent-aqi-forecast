"""Smoke tests around every rebuild and deploy (run by deploy_site.sh).

    .venv/bin/python -m aqf.smoke build   # before deploying: outputs are complete and sane
    .venv/bin/python -m aqf.smoke live    # after deploying: the public site serves this build

Only problems are printed, as `smoke FAIL: …` or `smoke WARN: …`, then one summary line.
FAIL means the build must not be published (for `live`: the public site is broken); WARN is
worth reporting but stops nothing. Exit code 1 on any FAIL.
"""
from __future__ import annotations

import fnmatch
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from .data import PROC, ROOT

OUT = ROOT / "output"
SITE = OUT / "site"
LOG = ROOT / "data" / "forecast_log"
URL = "https://tashkent-aqi.vercel.app"
DEPLOYABLE = {"index.html", "api/latest.js", "vercel.json"}  # everything else in output/site is ignored or wrong
NOT_PUBLIC = ["/README.md", "/run_daily.sh", "/requirements.txt", "/aqf/forecast.py", "/.env.local"]
DATA_RE = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)
INLINE_JS_RE = re.compile(r"<script(?![^>]*\bsrc=)(?![^>]*application/json)[^>]*>(.*?)</script>", re.S)


class Checks:
    def __init__(self, stage: str):
        self.stage, self.n, self.fails, self.warns = stage, 0, 0, 0

    def ok(self, cond, msg: str, warn: bool = False) -> bool:
        self.n += 1
        if not cond:
            print(f"smoke {'WARN' if warn else 'FAIL'}: {msg}", flush=True)
            if warn:
                self.warns += 1
            else:
                self.fails += 1
        return bool(cond)

    def done(self) -> int:
        print(f"smoke {self.stage}: {'FAILED' if self.fails else 'passed'} "
              f"({self.n} checks, {self.fails} failed, {self.warns} warnings)")
        return 1 if self.fails else 0


def _num(v, lo: float = 0, hi: float = 1000) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v) and lo <= v <= hi


def _js_ok(code: str) -> tuple[bool | None, str]:
    """Syntax check with `node --check`; None when node is not installed."""
    node = shutil.which("node") or next((p for p in ("/opt/homebrew/bin/node", "/usr/local/bin/node")
                                        if shutil.which(p)), None)
    if node is None:
        return None, ""
    with tempfile.NamedTemporaryFile("w", suffix=".js") as f:
        f.write(code)
        f.flush()
        r = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
    lines = r.stderr.strip().splitlines()
    return r.returncode == 0, next((x for x in lines if "Error" in x), lines[0] if lines else "")


def _page_data(html: str) -> dict | None:
    m = DATA_RE.search(html)
    try:
        return json.loads(m.group(1)) if m else None
    except ValueError:
        return None


# ---------------------------------------------------------------- before deploying

def check_forecast(c: Checks) -> dict | None:
    try:
        fc = json.loads((OUT / "forecast.json").read_text())
        origin, gen = pd.Timestamp(fc["origin"]), pd.Timestamp(fc["generated_at"])
    except Exception as e:  # noqa: BLE001
        c.ok(False, f"output/forecast.json unreadable ({type(e).__name__}: {e})")
        return None
    now = pd.Timestamp.now(tz="Asia/Tashkent")
    c.ok(now - gen < pd.Timedelta(hours=30), f"forecast.json is old (generated {gen:%Y-%m-%d %H:%M})", warn=True)
    c.ok(_num(fc.get("data_age_hours"), hi=6), f"latest observation is {fc.get('data_age_hours')} h old", warn=True)
    c.ok(not fc.get("warnings"), "data sources: " + "; ".join(fc.get("warnings") or []), warn=True)
    c.ok(_num(fc.get("latest_aqi")), f"latest_aqi is {fc.get('latest_aqi')!r}")
    trained = pd.Timestamp(fc.get("models_trained_at", "1970-01-01"))
    c.ok(now.tz_localize(None) - trained < pd.Timedelta(days=14), f"models last trained {trained:%Y-%m-%d}", warn=True)
    from .models import FIRST_VERSION, MODEL_VERSION
    ver = fc.get("model_version", FIRST_VERSION)
    c.ok(ver == MODEL_VERSION, f"forecast issued by model version {ver}, the code is at {MODEL_VERSION} "
         "(the next morning run issues with it, retraining first if models.pkl is older)", warn=True)

    first_day = {origin.normalize(), (origin + pd.Timedelta(hours=1)).normalize()}
    for key, n, step in (("hourly_24h", 24, "h"), ("hourly_7d", 168, "h"), ("daily_90d", 90, "D")):
        rows = fc.get(key) or []
        if not c.ok(len(rows) == n, f"{key}: {len(rows)} rows, expected {n}"):
            continue
        t = pd.DatetimeIndex([r["t"] for r in rows])
        starts = t[0] == origin + pd.Timedelta(hours=1) if step == "h" else t[0] in first_day
        c.ok(starts and (t[1:] - t[:-1] == pd.Timedelta(1, step)).all(),
             f"{key}: times do not run in steps of 1 {step} from the forecast origin {origin}")
        bad = [r["t"] for r in rows
               if not (all(_num(r.get(k)) for k in ("aqi", "lo", "hi")) and r["lo"] <= r["aqi"] <= r["hi"])]
        c.ok(not bad, f"{key}: {len(bad)} rows with a missing or out-of-range value or band (first {bad[:1]})")
        c.ok(len({r["aqi"] for r in rows}) > 1, f"{key}: the forecast is a constant")
    if len(fc.get("hourly_7d") or []) == 168:
        c.ok(fc.get("hourly_24h") == fc["hourly_7d"][:24], "hourly_24h differs from the first day of hourly_7d")
    c.ok(all(_num(r.get("p_red"), hi=1) and _num(r.get("p_purple"), hi=1) for r in fc.get("daily_90d") or []),
         "daily_90d: red/purple-day chance missing or outside 0–1")
    from .live import issue_day, issued_stems  # the network in the shadow (aqf.nn shadow), run_daily.sh
    stem = issued_stems()[-1:]
    if stem and issue_day(json.loads((LOG / f"{stem[0]}.json").read_text())) == issue_day(fc):
        c.ok((LOG / f"{stem[0]}_nn.json").exists(), f"shadow network forecast missing for today's forecast {stem[0]}", warn=True)
    return fc


def check_silam(c: Checks) -> None:
    """SILAM runs are copied daily (aqf.silam); FMI keeps only ~32 days, so a stalled copy is worth a warning."""
    runs = sorted(p.stem for p in (ROOT / "data" / "silam").glob("????????.parquet"))
    if not c.ok(runs, "no SILAM runs saved (aqf.silam fetch)", warn=True):
        return
    age = (pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None).normalize() - pd.Timestamp(runs[-1])).days
    c.ok(age <= 2, f"newest saved SILAM run is {runs[-1]} ({age} days old): copying fails, FMI keeps ~32 days", warn=True)


def check_satellite(c: Checks) -> None:
    """Sentinel-5P NO2 and FIRMS fires are copied daily (aqf.satellite); FIRMS keeps only 7 days openly."""
    base = ROOT / "data" / "satellite"
    today = pd.Timestamp.now(tz="Asia/Tashkent").tz_localize(None).normalize()
    for name, pattern, limit, why in (("NO2", "no2/????????_*.json", 3, "Sentinel-5P copy stalls"),
                                      ("FIRMS", "firms/????????.parquet", 4, "fires copy stalls, FIRMS keeps 7 days")):
        days = sorted(p.name[:8] for p in base.glob(pattern))
        if not c.ok(days, f"no {name} days saved (aqf.satellite fetch)", warn=True):
            continue
        age = (today - pd.Timestamp(days[-1])).days
        c.ok(age <= limit, f"newest saved {name} day is {days[-1]} ({age} days old): {why}", warn=True)


def check_models(c: Checks) -> None:
    p = PROC / "models.pkl"
    if c.ok(p.exists(), "data/processed/models.pkl is missing"):
        b = p.read_bytes()
        c.ok(b"lightgbm" in b and b"HistGradientBoosting" not in b,
             "models.pkl holds HistGradientBoosting instead of LightGBM (OpenMP fallback?)", warn=True)
    try:
        import numpy as np
        import lightgbm
        lightgbm.LGBMRegressor(n_estimators=2, min_child_samples=1, verbose=-1).fit(
            np.arange(20.0).reshape(-1, 1), np.arange(20.0))
    except Exception as e:  # noqa: BLE001
        c.ok(False, f"LightGBM does not run ({type(e).__name__}); the next retrain would fall back to HGB", warn=True)


def check_backtest(c: Checks) -> None:
    try:
        head = json.loads((PROC / "backtest_summary.json").read_text())["headline"]
        bands = json.loads((PROC / "intervals.json").read_text())
    except Exception as e:  # noqa: BLE001
        c.ok(False, f"backtest_summary.json / intervals.json unreadable ({type(e).__name__})")
        return
    for k, hi in (("h3", 40), ("h24", 60), ("h168", 80), ("d17", 60), ("d890", 80)):
        mae = head.get(k, {}).get("mae")
        c.ok(_num(mae, 1, hi), f"backtest {k}: MAE {mae!r} is not plausible")
    c.ok(bands.get("hourly") and bands.get("daily")
         and all(b["q10"] < 0 < b["q90"] for k in ("hourly", "daily") for b in bands[k]),
         "intervals.json: 80% bands missing or inverted")


def check_page(c: Checks, path, fc: dict | None, vercel: bool) -> None:
    rel = path.relative_to(OUT.parent).as_posix()
    html = path.read_text() if path.exists() else ""
    if not c.ok(len(html) > 50_000, f"{rel}: missing or too small ({len(html)} bytes)"):
        return
    data = _page_data(html)
    if c.ok(data and {"meta", "obs", "h", "d", "hist", "bt", "live"} <= data.keys(), f"{rel}: embedded data broken"):
        c.ok(len(data["h"]) == 168 and len(data["d"]) == 90 and data["hist"].get("aqi"),
             f"{rel}: embedded forecast or history is incomplete")
        if fc:
            c.ok(data["meta"]["origin"] == fc["origin"],
                 f"{rel}: built from forecast {data['meta']['origin']}, forecast.json is {fc['origin']}")
        c.ok(not data["live"].get("error"), f"{rel}: live accuracy tracking failed ({data['live'].get('error')})", warn=True)
        b = data["live"].get("bias") or {}
        if not vercel:  # once per build, not for both copies of the page
            c.ok(not b.get("alarm"), f"forecast drift: the next-day forecast was {b.get('mean', 0):+.0f} AQI off on average over "
                 f"the last {b.get('days')} scored days ({b.get('over')} too high, {b.get('under')} too low)", warn=True)
        why, daily = data.get("why") or [], {r[0][:10]: r[1] for r in data["d"]}
        c.ok(why, f"{rel}: forecast explanation (aqf.explain) missing", warn=True)
        c.ok(all(abs(w["now"] + sum(v for k, v in w.items() if k not in ("date", "now", "forecast")) - w["forecast"]) < 0.6
                 and abs(w["forecast"] - daily.get(w["date"], -99)) <= 1 for w in why),
             f"{rel}: forecast explanation does not add up to the daily forecast")
        rw = data.get("red_week") or {}
        c.ok(len(rw.get("marked_by_day") or []) == 7 and _num(rw.get("red_when_marked"), hi=1),
             f"{rel}: how the week's red-day mark did on past winters (aqf.report.red_week) missing", warn=True)
        mv = data.get("model") or {}
        c.ok(mv.get("versions") and not mv.get("error"), f"{rel}: model versions (what changed, since when) missing", warn=True)
        sc = (data.get("scen") or {}).get("scenarios") or []
        c.ok(len(sc) == 2 and all(x.get("days") and x.get("winters") for x in sc),
             f"{rel}: emission-cut scenarios (aqf.scenario) missing or incomplete", warn=True)
        attr = data.get("attr") or {}
        c.ok(attr.get("backtest"), f"{rel}: error breakdown (aqf.attribution) missing", warn=True)
        c.ok(all(abs(r["error"] - r["weather"] - r["pollution"] - r["unknown"] - r.get("rerun", 0) - r.get("network", 0)) < 0.6
                 for r in attr.get("live", [])), f"{rel}: error breakdown parts do not add up to the error")
        # a forecast that saved its inputs and was re-run with its own model must reproduce itself
        c.ok(all(abs(r.get("rerun", 0)) < 3 for r in attr.get("live", []) if r.get("weather_saved") and r.get("same_model")),
             f"{rel}: error breakdown re-run differs from the issued forecast by 3+ AQI", warn=True)
    for s in ("Tashkent", "Research alpha"):
        c.ok(s in html, f"{rel}: text «{s}» is missing")
    for code in INLINE_JS_RE.findall(html):
        good, err = _js_ok(code)
        if good is None:
            c.ok(False, "node not found; page script syntax not checked", warn=True)
            break
        c.ok(good, f"{rel}: page script does not parse: {err}")
    if vercel:
        c.ok(html.startswith("<!doctype html>"), f"{rel}: document head missing")
        c.ok("api/latest" in html, f"{rel}: live-reading call missing")
        c.ok("/_vercel/insights/script.js" in html, f"{rel}: analytics script missing", warn=True)


def check_deploy_folder(c: Checks) -> None:
    ignore = (SITE / ".vercelignore").read_text().split() if (SITE / ".vercelignore").exists() else []
    c.ok(".env*" in ignore, "output/site/.vercelignore does not exclude .env* (a token would be published)")
    extra = []
    for p in sorted(SITE.rglob("*")):
        rel = p.relative_to(SITE).as_posix()
        if p.is_dir() or rel.startswith(".vercel/") or rel == ".vercelignore" \
                or any(fnmatch.fnmatch(p.name, pat) for pat in ignore):
            continue
        if rel not in DEPLOYABLE:
            extra.append(rel)
    c.ok(not extra, f"output/site holds files that would be published: {extra[:5]}")
    js = SITE / "api" / "latest.js"
    if c.ok(js.exists(), "output/site/api/latest.js is missing"):
        good, err = _js_ok(js.read_text())
        c.ok(good is not False, f"api/latest.js does not parse: {err}")
    try:
        vj = json.loads((SITE / "vercel.json").read_text())
    except Exception as e:  # noqa: BLE001
        c.ok(False, f"output/site/vercel.json invalid ({type(e).__name__})")
        vj = {}
    # the CSP must allow exactly the page's own inline scripts, or browsers block the page
    from .site import csp
    hdr = {h["key"]: h["value"] for g in vj.get("headers", []) for h in g.get("headers", [])}
    index = (SITE / "index.html").read_text() if (SITE / "index.html").exists() else ""
    c.ok(hdr.get("Content-Security-Policy") == csp(index), "vercel.json CSP does not match the page's inline scripts")
    c.ok(hdr.get("X-Content-Type-Options") == "nosniff" and "frame-ancestors 'none'" in hdr.get("Content-Security-Policy", ""),
         "vercel.json security headers missing")


def check_git(c: Checks) -> None:
    """Every code change should reach GitHub (data, outputs and logs are not tracked)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    st = git("status", "--porcelain")
    if not c.ok(st.returncode == 0, "git status failed; GitHub sync not checked", warn=True):
        return
    changed = [line[3:] for line in st.stdout.splitlines()]
    c.ok(not changed, f"code changes not committed to GitHub: {changed[:5]}", warn=True)
    ahead = git("rev-list", "--count", "@{u}..HEAD").stdout.strip()
    c.ok(ahead == "0", f"{ahead or '?'} commit(s) not pushed to GitHub", warn=True)


def build() -> int:
    c = Checks("build")
    check_git(c)
    fc = check_forecast(c)
    check_models(c)
    check_silam(c)
    check_satellite(c)
    check_backtest(c)
    check_page(c, OUT / "dashboard.html", fc, vercel=False)
    check_page(c, SITE / "index.html", fc, vercel=True)
    check_deploy_folder(c)
    return c.done()


# ---------------------------------------------------------------- after deploying

def live() -> int:
    import requests
    c = Checks("live")
    s = requests.Session()
    s.headers["user-agent"] = "tashkent-aqi smoke test"

    def get(path: str):
        try:
            return s.get(URL + path, timeout=20, allow_redirects=False)
        except requests.RequestException as e:
            print(f"smoke: {path}: {type(e).__name__}", flush=True)
            return None

    want = json.loads((OUT / "forecast.json").read_text())["origin"]
    origin = None
    for _ in range(6):  # the production alias can take a few seconds to switch
        r = get("/")
        data = _page_data(r.text) if r is not None and r.status_code == 200 else None
        origin = data["meta"]["origin"] if data else None
        if origin == want:
            break
        time.sleep(10)
    c.ok(r is not None and r.status_code == 200, f"{URL}/: HTTP {getattr(r, 'status_code', None)}")
    c.ok(origin == want, f"{URL} serves forecast {origin}, expected {want}")
    c.ok(r is not None and "Research alpha" in r.text, f"{URL}: alpha notice missing")

    r = get("/api/latest?fresh=1")
    code = getattr(r, "status_code", None)
    j = r.json() if code in (200, 502) else {}
    if code == 502:  # the function works, tashkentairdata.uz does not
        c.ok(False, f"/api/latest: upstream error {j.get('error')}", warn=True)
    elif c.ok(code == 200 and _num(j.get("aqi")), f"/api/latest: HTTP {code} {str(j)[:100]}"):
        age = pd.Timestamp.now(tz="Asia/Tashkent") - pd.Timestamp(j["recorded_at"])
        c.ok(age < pd.Timedelta(hours=6), f"/api/latest: reading is {age.total_seconds() / 3600:.1f} h old", warn=True)

    r = get("/")
    h = r.headers if r is not None else {}
    c.ok("frame-ancestors 'none'" in h.get("content-security-policy", "") and h.get("x-content-type-options") == "nosniff",
         "security headers are not served (CSP, nosniff)")
    r = get("/api/latest?fresh=1")
    c.ok(r is not None and "no-store" not in r.headers.get("cache-control", "") or getattr(r, "status_code", 0) != 200,
         "/api/latest?fresh=1 bypasses the edge cache")
    r = get("/_vercel/insights/script.js")
    c.ok(getattr(r, "status_code", None) == 200, "analytics script is not served", warn=True)
    # the repository must never be served (2026-10-18: a git-linked push published the code)
    for path in NOT_PUBLIC:
        r = get(path)
        c.ok(getattr(r, "status_code", None) != 200,
             f"{path} is publicly served: the repository may be deployed, check the Vercel git link")
    return c.done()


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"
    if stage not in ("build", "live"):
        sys.exit("usage: python -m aqf.smoke [build|live]")
    sys.exit(build() if stage == "build" else live())
