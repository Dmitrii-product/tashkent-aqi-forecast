"""Build the Vercel version of the site into output/site/.

The claude.ai artifact page runs in a sandbox that blocks requests to other hosts, so the
"latest reading" there is whatever the last forecast run saw. On Vercel the same page also
calls its own serverless function /api/latest, which reads the latest value on demand from LATEST_AQI_URL
(5-minute shared cache; the refresh button asks for a fresh value). The Vercel copy also
carries the Vercel Web Analytics script and security headers (a hash-based CSP and others).

Writes output/site/{index.html, api/latest.js, vercel.json}; run_daily.sh then deploys the
folder with the Vercel CLI (`vercel deploy --prod`, project linked in output/site/.vercel).

    .venv/bin/python -m aqf.site
"""
from __future__ import annotations

import base64
import hashlib
import json
import re

from .data import ROOT

SITE = ROOT / "output" / "site"
PROJECT = "tashkent-aqi"

# the same head the claude.ai artifact skeleton adds, so both copies render alike
HEAD = ('<!doctype html><html><head><meta charset=utf8><meta name=viewport '
        'content="width=device-width,initial-scale=1,viewport-fit=cover"><style>'
        ':root{color-scheme:light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);'
        'padding-bottom:env(safe-area-inset-bottom,0px)}html{scroll-padding-top:env(safe-area-inset-top,0px)}'
        'body{margin:0;padding:0;font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#fff;color:#000}'
        'img{max-width:100%}[hidden]{display:none!important}</style></head><body>')

# Vercel Web Analytics, plain-HTML form (no cookies); only in the Vercel copy, since the
# route /_vercel/insights/* exists only on Vercel deployments with analytics enabled
ANALYTICS = ('<script>window.va=window.va||function(){(window.vaq=window.vaq||[]).push(arguments)};</script>'
             '<script defer src="/_vercel/insights/script.js"></script>')

LATEST_JS = r"""// Latest Tashkent AQI (IQAir city data) from the URL in the LATEST_AQI_URL environment variable.
// Shared edge cache of 5 minutes; the page's refresh button (?fresh=1) gets a value at most
// 30 s old. A warm instance also keeps the upstream reply for 30 s, so cache-busting URLs
// cannot turn into a flood of requests to the source.
let memo = null;  // { at: ms, body: {...} }
module.exports = async (req, res) => {
  const fresh = req.query && req.query.fresh;
  try {
    if (!memo || Date.now() - memo.at > 30000) {
      if (!process.env.LATEST_AQI_URL) throw new Error('LATEST_AQI_URL is not set');
      const r = await fetch(process.env.LATEST_AQI_URL, {
        headers: { 'user-agent': 'tashkent-aqi-forecast (vercel function)' },
      });
      if (!r.ok) throw new Error('upstream HTTP ' + r.status);
      const j = await r.json();
      const latest = (j && j.latest) || {};
      const aqi = latest.aqi;
      if (typeof aqi !== 'number' || !isFinite(aqi) || aqi < 0 || aqi > 1000) throw new Error('no valid latest value upstream');
      const at = typeof latest.recorded_at === 'string' && !isNaN(Date.parse(latest.recorded_at)) ? latest.recorded_at.slice(0, 40) : null;
      memo = { at: Date.now(), body: { aqi, recorded_at: at, source: 'tashkentairdata.uz (IQAir)' } };
    }
    res.setHeader('Cache-Control', fresh ? 'public, max-age=0, s-maxage=30' : 'public, max-age=0, s-maxage=300, stale-while-revalidate=300');
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.status(200).send(JSON.stringify({ ...memo.body, fetched_at: new Date(memo.at).toISOString() }));
  } catch (e) {
    res.setHeader('Cache-Control', 'no-store');
    res.status(502).json({ error: 'upstream unavailable' });
  }
};
"""

VERCEL_JSON = {"cleanUrls": True, "functions": {"api/latest.js": {"maxDuration": 10}}}
INLINE_JS = re.compile(r"<script(?![^>]*\bsrc=)(?![^>]*application/json)[^>]*>(.*?)</script>", re.S)


def csp(html: str) -> str:
    """Content-Security-Policy for the page: scripts only from this site and the page's own
    inline scripts, by hash (recomputed every build); styles inline and from Google Fonts;
    connections only back to this site (/api/latest, analytics); never framed elsewhere."""
    hashes = " ".join(f"'sha256-{base64.b64encode(hashlib.sha256(m.encode()).digest()).decode()}'"
                      for m in INLINE_JS.findall(html))
    return ("default-src 'self'; "
            f"script-src 'self' {hashes}; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def headers(html: str) -> list[dict]:
    return [{"source": "/(.*)", "headers": [
        {"key": "Content-Security-Policy", "value": csp(html)},
        {"key": "X-Content-Type-Options", "value": "nosniff"},
        {"key": "X-Frame-Options", "value": "DENY"},
        {"key": "Referrer-Policy", "value": "strict-origin-when-cross-origin"},
        {"key": "Permissions-Policy", "value": "camera=(), microphone=(), geolocation=(), payment=(), usb=()"},
    ]}]


def build() -> dict[str, str]:
    page = (ROOT / "output" / "dashboard.html").read_text()
    index = HEAD + page + ANALYTICS + "</body></html>"
    files = {
        "index.html": index,
        "api/latest.js": LATEST_JS,
        "vercel.json": json.dumps({**VERCEL_JSON, "headers": headers(index)}, indent=1),
    }
    for name, text in files.items():
        p = SITE / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (SITE / "deploy_files.json").unlink(missing_ok=True)
    return files


if __name__ == "__main__":
    f = build()
    print(SITE, {k: len(v) for k, v in f.items()})
