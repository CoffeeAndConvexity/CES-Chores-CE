#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

# Reproduce the bidding experiment grid.
for rho in 5.0 2.0 1.2; do
    "$python_bin" "$repo_root/code/adaptive_stepsize_bidding.py" \
        --rho "$rho" --max-steps 10000 --N 100 --M 500 \
        --noise-level 1 --num-instances 10 "$@"
done
