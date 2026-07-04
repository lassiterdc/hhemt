#!/usr/bin/env bash
# Idempotently apply the committed main-branch ruleset to lassiterdc/hhemt.
# list-then-PUT-or-POST: create on first run, update on subsequent runs
# (POST /rulesets is NOT idempotent; a duplicate POST creates a second ruleset).
# Reads the LOCAL committed JSON, so apply is decoupled from push/branch state.
set -euo pipefail

OWNER="lassiterdc"
REPO="hhemt"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
RULESET_JSON="${REPO_ROOT}/.github/rulesets/main.json"

[ -f "$RULESET_JSON" ] || { echo "ERROR: ruleset JSON not found at $RULESET_JSON" >&2; exit 1; }
NAME="$(jq -r '.name' "$RULESET_JSON")"

existing_id="$(gh api "/repos/${OWNER}/${REPO}/rulesets" \
  --jq ".[] | select(.name == \"${NAME}\") | .id")"

if [ -n "${existing_id}" ]; then
  echo "Updating existing ruleset '${NAME}' (id=${existing_id}) via PUT..."
  gh api --method PUT \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/${OWNER}/${REPO}/rulesets/${existing_id}" \
    --input "$RULESET_JSON"
else
  echo "Creating ruleset '${NAME}' via POST..."
  gh api --method POST \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "/repos/${OWNER}/${REPO}/rulesets" \
    --input "$RULESET_JSON"
fi

echo "Active rulesets on ${OWNER}/${REPO}:"
gh api "/repos/${OWNER}/${REPO}/rulesets" --jq '.[] | {id, name, enforcement, target}'
