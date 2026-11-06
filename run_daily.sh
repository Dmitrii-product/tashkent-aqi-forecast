#!/bin/zsh
# Daily update: refresh data, retrain if models are older than 7 days, rebuild the dashboard.
# Writes output/forecast.json, output/forecast_*.csv, output/dashboard.html; log in logs/daily.log.
set -euo pipefail
cd "${0:A:h}"
# LightGBM needs an OpenMP runtime; use the copy bundled with scikit-learn (no system install)
export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')"
  .venv/bin/python -W ignore -m aqf.forecast
  # SILAM (FMI) forecasts for Tashkent: each new run copied once and kept (FMI keeps ~32 days)
  .venv/bin/python -W ignore -m aqf.silam fetch || echo "WARNING: SILAM fetch failed"
  .venv/bin/python -W ignore -m aqf.satellite fetch || echo "WARNING: satellite fetch failed"
  # the neural network in the shadow, scored on the other-forecasts tab; its own process (PyTorch),
  # and a failure must not stop the forecast from going out
  .venv/bin/python -W ignore -m aqf.nn shadow || echo "WARNING: shadow network forecast failed"
  .venv/bin/python -m aqf.dashboard
  # smoke tests, public site on Vercel, smoke tests of the live site
  ./deploy_site.sh
  echo "ok"
} >> logs/daily.log 2>&1
tail -n 8 logs/daily.log
