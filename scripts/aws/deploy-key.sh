#!/usr/bin/env bash
# A read-only GitHub deploy key for this machine (runbook step 4). Run ON the instance. Idempotent.
#
#   bash deploy-key.sh
#
# The private key never leaves the machine. Only the public line is printed, for a founder to add
# under the repository's Settings -> Deploy keys with write access unticked (or with
# `gh repo deploy-key add KEY.pub --repo az5arpeera/NeurOS --title neuros-aws`, read-only by
# default). GitHub's host keys come from its HTTPS API, not trusted on first use.
set -euo pipefail

KEY="$HOME/.ssh/github_deploy"
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [ ! -f "$KEY" ]; then
  ssh-keygen -q -t ed25519 -N "" -C "neuros-aws-deploy" -f "$KEY"
fi

# Use this key, and only this key, for github.com.
if ! grep -qs "IdentityFile ~/.ssh/github_deploy" "$HOME/.ssh/config"; then
  printf 'Host github.com\n    IdentityFile ~/.ssh/github_deploy\n    IdentitiesOnly yes\n' \
    >> "$HOME/.ssh/config"
fi
chmod 600 "$HOME/.ssh/config"

if ! grep -qs '^github.com ' "$HOME/.ssh/known_hosts"; then
  keys="$(curl -fsSL https://api.github.com/meta \
    | python3 -c 'import json, sys; print("\n".join(json.load(sys.stdin)["ssh_keys"]))')"
  [ -n "$keys" ] || { echo "ERROR: no host keys from api.github.com/meta" >&2; exit 1; }
  while IFS= read -r key; do
    printf 'github.com %s\n' "$key"
  done <<< "$keys" >> "$HOME/.ssh/known_hosts"
fi

echo "--- GitHub host key fingerprints now trusted (compare with GitHub's documentation):"
ssh-keygen -lf "$HOME/.ssh/known_hosts" | grep -i github
echo "--- public deploy key (add on GitHub, read-only):"
cat "$KEY.pub"
# `ssh -T git@github.com` exits 1 even on success (GitHub offers no shell): match its text.
# </dev/null: when this script itself arrives on stdin (ssh host 'bash -s' < script), ssh would
# otherwise swallow the rest of it.
greeting="$(ssh -o BatchMode=yes -T git@github.com </dev/null 2>&1 || true)"
if [[ "$greeting" == *"successfully authenticated"* ]]; then
  echo "GitHub already accepts this key: $greeting"
else
  echo "GitHub doesn't accept the key yet (add it, then clone): $greeting"
fi
