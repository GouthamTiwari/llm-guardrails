#!/bin/sh
# Audit: every commit reachable from any ref must be authored/committed by Goutham Tiwari
# with no co-author lines. Run before pushing and before releases.
cd "$(git rev-parse --show-toplevel)" || exit 1
. "$(dirname "$0")/_check.sh"
n=$(git rev-list --all --count)
if check_range --all; then
  echo "OK: all $n commits authored by $OWNER_NAME, no co-author lines"
else
  echo "FAIL: see messages above. Fix the commits before pushing." >&2
  exit 1
fi
