#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [commit message]\n' "$0" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  commit_message=$1
else
  read -r -p 'Commit message: ' commit_message
fi

if [[ -z ${commit_message//[[:space:]]/} ]]; then
  printf 'Commit message cannot be empty.\n' >&2
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

git add -A
if git diff --cached --quiet; then
  printf 'No changes to commit.\n'
  exit 0
fi

git commit -m "$commit_message"
git push
