#!/usr/bin/env bash
#
# lock.sh - regenerate the lock file for THIS platform from requirements/*.in.
#
# Resolution is platform specific, so a lock can only be produced on a machine of
# that platform. Run it on each and commit both.
#
# Needs pip-tools (requirements/dev.in). Uses uv when it is present because it is
# faster and reads the same files.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Run from the repo root and use relative paths: uv and pip-compile both record
# the command they were invoked with in the file header, and absolute paths there
# would carry one machine's home directory into the repository.
cd "$REPO"

case "$(uname -s)" in
    Linux)  OUT="requirements/lock/linux.txt" ;;
    Darwin) OUT="requirements/lock/macos.txt" ;;
    *)      echo "Unsupported platform: $(uname -s)" >&2; exit 1 ;;
esac

# Every runtime slice goes into one lock. The slices exist so a machine can install
# a subset from the .in files; the lock is the full, reproducible set.
INPUTS=("requirements/arm.in" "requirements/sim.in" "requirements/umi.in")

echo "==> locking $(basename "$OUT") from: ${INPUTS[*]}"

if command -v uv >/dev/null 2>&1; then
    uv pip compile --quiet --output-file "$OUT" "${INPUTS[@]}"
elif command -v pip-compile >/dev/null 2>&1; then
    pip-compile --quiet --strip-extras --output-file "$OUT" "${INPUTS[@]}"
else
    echo "Neither uv nor pip-compile found." >&2
    echo "  pip install -r requirements/dev.in" >&2
    exit 1
fi

echo "==> wrote $OUT  ($(grep -cv '^[[:space:]]*#\|^[[:space:]]*$' "$OUT") pinned packages)"
echo "    Commit it. A lock that is not committed is not a lock."
