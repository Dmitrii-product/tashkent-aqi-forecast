#!/bin/zsh
# Monthly maintenance: backtest refresh, 80% band check against live errors, report, site.
# Takes 30–60 min (the backtest). Log: logs/monthly.log; report: logs/monthly_report_<YYYY-MM>.md
set -euo pipefail
cd "${0:A:h}"
{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S')"
  .venv/bin/python -W ignore -m aqf.monthly "$@"
  # smoke tests, public site on Vercel, smoke tests of the live site
  ./deploy_site.sh
  echo "ok"
} >> logs/monthly.log 2>&1
tail -n 40 logs/monthly.log
