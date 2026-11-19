#!/bin/sh
# Shared checks for authorship (CLAUDE.md Rule 0). Sourced by the other hooks.
# Only author and committer allowed: Goutham Tiwari <227244284+GouthamTiwari@users.noreply.github.com>

OWNER_NAME="Goutham Tiwari"
OWNER_EMAIL="227244284+GouthamTiwari@users.noreply.github.com"

# Lines that must never appear in a commit message.
FORBIDDEN_MSG='^co-authored-by:|^claude-session:|generated with \[?claude|claude\.ai/code|noreply@anthropic\.com|^🤖'

# Checks a message file or stdin text. Prints the offending lines, returns 1 if any.
check_message_text() {
  bad=$(printf '%s\n' "$1" | grep -iE "$FORBIDDEN_MSG")
  if [ -n "$bad" ]; then
    echo "Blocked: commit message has an attribution line (CLAUDE.md Rule 0):" >&2
    printf '%s\n' "$bad" >&2
    return 1
  fi
  # Signed-off-by is allowed only for the owner.
  so=$(printf '%s\n' "$1" | grep -iE '^signed-off-by:' | grep -vF "$OWNER_NAME <$OWNER_EMAIL>")
  if [ -n "$so" ]; then
    echo "Blocked: Signed-off-by must be $OWNER_NAME <$OWNER_EMAIL> only:" >&2
    printf '%s\n' "$so" >&2
    return 1
  fi
  return 0
}

# Checks every commit in a git revision range (arguments are passed to git log).
# Returns 1 if any commit has the wrong author/committer or a forbidden message line.
check_range() {
  fail=0
  for sha in $(git rev-list "$@"); do
    ident=$(git log -1 --format='%an <%ae>|%cn <%ce>' "$sha")
    author=${ident%%|*}
    committer=${ident#*|}
    want="$OWNER_NAME <$OWNER_EMAIL>"
    if [ "$author" != "$want" ] || [ "$committer" != "$want" ]; then
      echo "Blocked: commit $(git rev-parse --short "$sha") has author '$author' / committer '$committer' (want '$want')" >&2
      fail=1
    fi
    msg=$(git log -1 --format='%B' "$sha")
    check_message_text "$msg" >/dev/null 2>&1 || {
      echo "Blocked: commit $(git rev-parse --short "$sha") message contains an attribution/trailer line" >&2
      fail=1
    }
  done
  return $fail
}
