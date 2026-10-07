#!/usr/bin/env bash
# Usage: ./scripts/push_to_github.sh git@github.com:<you>/oci-genai-professional-certification.git
# Create the EMPTY repo on GitHub first (no README/licence/.gitignore), then run this from the repo root.
set -euo pipefail

REMOTE="${1:-}"
if [[ -z "$REMOTE" ]]; then
  echo "usage: $0 <remote-url>"; exit 1
fi
command -v git >/dev/null || { echo "git is not installed"; exit 1; }

# Safety net: never push secrets
if git ls-files --others --cached --exclude-standard 2>/dev/null | grep -Eq '(^|/)(\.env|.*\.pem|oci_config)$'; then
  echo "Refusing to continue: a secret file (.env / .pem / oci_config) is about to be committed."; exit 1
fi

[[ -d .git ]] || git init -b main
git add -A
echo "$(git status --short | wc -l | tr -d " ") changed/new files staged"
git diff --cached --quiet || git commit -m "feat: OCI Generative AI Professional certification labs, notes and practice exams"
git branch -M main
git remote get-url origin >/dev/null 2>&1 && git remote set-url origin "$REMOTE" || git remote add origin "$REMOTE"
git push -u origin main
echo "Done: $REMOTE"
