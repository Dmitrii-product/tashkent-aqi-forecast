#!/bin/zsh
# Build output/site/ from output/dashboard.html, smoke-test it, deploy it to Vercel and smoke-test
# the public site. Called by run_daily.sh and run_monthly.sh; run it by hand after code changes.
# A failed build test stops here (exit 1, nothing deployed, yesterday's site stays online).
set -euo pipefail
cd "${0:A:h}"
.venv/bin/python -m aqf.site
.venv/bin/python -W ignore -m aqf.smoke build
# project linked in output/site/.vercel; the CLI was logged in once by the user
VERCEL=$(command -v vercel || echo "$PWD/tools/node_modules/.bin/vercel")
if (cd output/site && "$VERCEL" deploy --prod --yes > ../../logs/vercel_deploy.log 2>&1); then
  echo "vercel: deployed https://tashkent-aqi.vercel.app"
  .venv/bin/python -W ignore -m aqf.smoke live || echo "WARNING: live smoke test failed (see smoke FAIL lines)"
else
  echo "WARNING: vercel deploy failed, see logs/vercel_deploy.log"
fi
