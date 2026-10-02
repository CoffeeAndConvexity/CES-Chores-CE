"""Compare adaptive-step-size updates on the bundled Spliddit matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import (
    DEFAULT_OUTPUT_DIR,
    METHODS,
    REPO_ROOT,
    RunResult,
    append_result,
    ces_rho,
    compute_update as _compute_update,
    positive_int,
    run_instance,
    search_stepsize,
    validate_instance,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for a stable, fast step size on Spliddit instances."
    )
    parser.add_argument("--rho", type=ces_rho, default=2.0)
    parser.add_argument("--max-steps", type=positive_int, default=10000)
    parser.add_argument(
        "--first-instances",
        type=int,
        default=20,
        help="Number of instances to run; -1 runs all instances.",
    )
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=REPO_ROOT / "data" / "distribute_tasks_valuation_matrices",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if args.first_instances != -1 and args.first_instances < 1:
        parser.error("--first-instances must be positive or -1")
    if not args.matrix_dir.is_dir():
        parser.error(f"matrix directory does not exist: {args.matrix_dir}")
    return args


def instance_files(matrix_dir: Path) -> list[Path]:
    """Use index order, then sorted unindexed files; ignore metadata CSVs.

    Historical results used filesystem enumeration, whose order was unspecified.
    The matrix_file output column provides a stable identity across copies.
    """
    available = {
        path.name: path
        for path in matrix_dir.glob("*.csv")
        if not path.name.startswith("_")
    }
    ordered = []
    index_path = matrix_dir / "_index.csv"
    if index_path.exists():
        index = pd.read_csv(index_path)
        if "matrix_file" not in index:
            raise ValueError(f"{index_path} must contain a matrix_file column")
        for name in index["matrix_file"]:
            if name in available:
                ordered.append(available.pop(name))
    ordered.extend(available[name] for name in sorted(available))
    if not ordered:
        raise ValueError(f"No instance matrices found in {matrix_dir}")
    return ordered


def run(
    instance: tuple[np.ndarray, np.ndarray],
    method: str,
    eta: float,
    max_steps: int,
    *,
    rho: float = 2.0,
) -> RunResult:
    return run_instance(*instance, method, eta, max_steps, rho)


def search_largest_stepsize(
    instance: tuple[np.ndarray, np.ndarray],
    method: str = "multiplicative",
    max_steps: int = 10000,
    *,
    rho: float = 2.0,
) -> tuple[float, dict]:
    return search_stepsize(
        lambda eta: run(instance, method, eta, max_steps, rho=rho),
        max_steps,
        verbose=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    files = instance_files(args.matrix_dir)
    if args.first_instances != -1:
        files = files[: args.first_instances]
    for method in METHODS:
        for instance_idx, path in enumerate(files):
            print(f"Testing {method} on {path.name}")
            disutilities = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
            budgets = np.ones(disutilities.shape[0])
            validate_instance(disutilities, budgets, args.rho)
            best_eta, data = search_largest_stepsize(
                (disutilities, budgets),
                method,
                args.max_steps,
                rho=args.rho,
            )
            append_result(
                args.output_dir,
                f"batch_adaptive_stepsize_{method}_spliddit.csv",
                {
                    "method": method,
                    "instance_idx": instance_idx,
                    "rho": args.rho,
                    "eta": best_eta,
                    **data,
                    "matrix_file": path.name,
                },
            )


if __name__ == "__main__":
    main()
