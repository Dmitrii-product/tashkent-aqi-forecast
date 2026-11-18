#!/bin/zsh
# Daily update: refresh data, retrain if models are older than 7 days, rebuild the dashboard.
# Writes output/forecast.json, output/forecast_*.csv, output/dashboard.html; log in logs/daily.log.
set -euo pipefail
cd "${0:A:h}"
# LightGBM needs an OpenMP runtime; use the copy bundled with scikit-learn (no system install)
export DYLD_FALLBACK_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
# One run at a time: on 2026-11-12 two catch-up runs started 3 s apart and raced on the same files.
# The lock is a directory (mkdir is atomic) holding the owner's pid; a lock whose owner is gone is stale.
LOCK=logs/.run_daily.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  other=$(cat "$LOCK/pid" 2>/dev/null || true)
  if [[ -n "$other" ]] && ps -p "$other" -o command= 2>/dev/null | grep -q run_daily; then
    alive=1
  elif [[ -z "$other" && -z "$(find "$LOCK" -maxdepth 0 -mmin +1)" ]]; then
    alive=1  # just created, pid not written yet
  else
    alive=0
  fi
  if (( alive )); then
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') skipped: another run_daily.sh is running (pid ${other:-?})" >> logs/daily.log
    echo "skipped: another run_daily.sh is running (pid ${other:-?})"
    exit 0
  fi
  rm -rf "$LOCK"
  mkdir "$LOCK" 2>/dev/null || { echo "skipped: another run_daily.sh took the lock"; exit 0; }
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
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
