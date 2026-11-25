#!/usr/bin/env bash
# Run Python with an existing Isaac Sim environment, but load THIS repo's code.
#
# This repo has no .venv of its own yet (`uv sync` downloads several GB). The venv from
# another checkout of the same codebase works, except that its editable install maps the
# `residual_copilot` package to that checkout's source/. A symlink earlier on PYTHONPATH
# takes precedence over the editable install, so imports resolve here instead.
#
# Usage (from anywhere):
#   scripts/isaac_python.sh scripts/play.py --task TubeTransfer --pilot SpaceMousePilot --two_cam
# Override the environment with ISAAC_VENV=/path/to/.venv.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ISAAC_VENV:-$HOME/Residual_Copilot/.venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
    echo "No Python at $VENV/bin/python. Set ISAAC_VENV to an Isaac Sim 5.1 / Isaac Lab 2.3 venv." >&2
    exit 1
fi

SHIM="$REPO/.pypath"
mkdir -p "$SHIM"
ln -sfn "$REPO/source" "$SHIM/residual_copilot"

export PYTHONPATH="$SHIM${PYTHONPATH:+:$PYTHONPATH}"
# Isaac Sim can exhaust static TLS slots ("cannot allocate memory in static TLS block").
export LD_PRELOAD="/lib/x86_64-linux-gnu/libgomp.so.1${LD_PRELOAD:+:$LD_PRELOAD}"

cd "$REPO"
exec "$VENV/bin/python" "$@"
