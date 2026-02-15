#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Assume we're running from the build directory (CTest sets WORKING_DIRECTORY)
WEBSERVER="${PWD}/bin/webserver"

# Pick interpreter: prefer project .venv, else system python3
PY_VENV="$PROJECT_ROOT/.venv/bin/python"
if [[ -x "$PY_VENV" ]]; then
  PY="$PY_VENV"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "python3 not found"; exit 1
fi

# Require 'requests` library 
"$PY" - <<'PY'
import sys, importlib.util
sys.exit(0 if importlib.util.find_spec("requests") else 2)
PY
status=$?
if [ "$status" -ne 0 ]; then
  echo "Missing 'requests'. Use the class dev container or create a venv:"
  echo "  python3 -m venv .venv && . .venv/bin/activate && pip install -r tests/integration/requirements.txt"
  exit 1
fi

[[ -f "$WEBSERVER" ]] || {
  echo "Webserver binary not found at $WEBSERVER"
  echo "Build: mkdir -p build && cd build && cmake .. && make"
  exit 1
}

echo "Running integration tests with $PY"
exec "$PY" "$SCRIPT_DIR/integration_test.py" --binary "$WEBSERVER" "$@"