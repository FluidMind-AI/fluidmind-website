#!/usr/bin/env bash
# Manual deploy of fluidmind.ai to the shared FluidMind droplet (Caddy static
# host, 161.35.12.67). No build step — this is a plain static site, so deploy is
# just an rsync of the repo files to /var/www/fluidmind.ai. Caddy serves them
# immediately; zero downtime.
#
#   ./scripts/deploy-droplet.sh        # sync repo -> droplet
#   DRY=1 ./scripts/deploy-droplet.sh  # preview what would change
#
# Requires the deploy key (~/.ssh/vargas_deploy_key) — the `deploy` user that
# owns the site's files on the box.
set -euo pipefail

HOST="deploy@161.35.12.67"
DEST="/var/www/fluidmind.ai/"
KEY="${DEPLOY_KEY:-$HOME/.ssh/vargas_deploy_key}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "▸ Syncing static site → ${HOST}:${DEST}"
rsync -az --delete ${DRY:+--dry-run -i} \
  --exclude='.git' --exclude='.gitignore' --exclude='CNAME' \
  --exclude='deploy' --exclude='scripts' --exclude='*.md' \
  -e "ssh -i '$KEY' -o BatchMode=yes" \
  ./ "${HOST}:${DEST}"

[[ -n "${DRY:-}" ]] && { echo "▸ Dry run only — nothing changed."; exit 0; }

echo "▸ Verifying over HTTPS…"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://fluidmind.ai/ || echo 000)
if [[ "$code" == "200" ]]; then
  echo "✓ Live: https://fluidmind.ai/ ($code)"
else
  scode=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 https://fluidmind.161.35.12.67.nip.io/ || echo 000)
  echo "• fluidmind.ai returned $code (expected until DNS points at the droplet)."
  echo "  Staging host https://fluidmind.161.35.12.67.nip.io/ → $scode"
fi
