#!/usr/bin/env bash
# Mandatory milestone check: lint, the complete fast suite, the integration suite and the web
# type-check and build. The Stop hook runs only the smoke subset; this script is what a milestone
# must pass before review. Usage: scripts/milestone_check.sh [output file]
set -uo pipefail
cd "$(dirname "$0")/.."
out="${1:-/dev/stdout}"
status=0
run() {
  local name="$1"; shift
  echo "### $name" >>"$out"
  local start=$SECONDS
  if "$@" >>"$out" 2>&1; then echo "PASS $name ($((SECONDS - start)) s)" >>"$out"; else echo "FAIL $name ($((SECONDS - start)) s)" >>"$out"; status=1; fi
}
[ "$out" != /dev/stdout ] && : >"$out"
echo "## $(date -u +%Y-%m-%dT%H:%M:%SZ) git $(git rev-parse --short HEAD) dirty=$(test -n "$(git status --porcelain)" && echo yes || echo no)" >>"$out"
run "ruff check" uv run --frozen ruff check .
run "ruff format --check" uv run --frozen ruff format --check .
run "smoke subset (Stop hook)" uv run --frozen --offline pytest -q -m 'smoke and not integration' -p no:cacheprovider
run "complete fast suite" uv run --frozen --offline pytest -q -m 'not integration' -p no:cacheprovider
run "integration suite" uv run --frozen pytest -q -m integration -p no:cacheprovider
run "web typecheck" bash -c "cd web && npm run typecheck"
run "web build" bash -c "cd web && npm run build"
rm -f MUJOCO_LOG.TXT
echo "milestone check: $([ $status -eq 0 ] && echo PASS || echo FAIL)" >>"$out"
exit $status
