#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

# Compatibility wrapper. Prefer invoking scripts/run.py directly so every
# setting is visibly expressed as a Hydra override.
exec "${HERMES_PYTHON:-python}" scripts/run.py \
    run.num_chunks=8 run.sample_fps=0.5 run.kv_size=6000 "$@"
