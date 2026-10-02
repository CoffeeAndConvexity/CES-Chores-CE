#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

# Reproduce the synthetic integer-uniform experiment grid.
for n_resources in 500 1000 2000; do
    for rho in 1.2 2.0 5.0; do
        "$python_bin" "$repo_root/code/adaptive_stepsize.py" \
            --n-agents 100 --n-resources "$n_resources" --rho "$rho" \
            --num-instances 20 --max-steps 10000 \
            --value-distribution integer_uniform "$@"
    done
done
