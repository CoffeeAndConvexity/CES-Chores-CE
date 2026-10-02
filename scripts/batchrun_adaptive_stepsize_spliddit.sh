#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

# Reproduce the experiment grid over all Spliddit matrices.
for rho in 1.2 2.0 5.0; do
    "$python_bin" "$repo_root/code/adaptive_stepsize_spliddit.py" \
        --rho "$rho" --max-steps 60000 --first-instances -1 "$@"
done
