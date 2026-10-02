"""Shared mechanics for the adaptive-step-size experiments.

The experiment updates deliberately use raw excess supply, matching the original
experiments. Convergence is recorded at the zero-based iteration before its update.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from ces_disutility import compute_excess_supply

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "results"
METHODS = ("multiplicative", "additive")
EPSILONS = (0.1, 0.05, 0.01, 0.005, 0.001)
RunResult = tuple[bool, dict | str]


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def ces_rho(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 1:
        raise argparse.ArgumentTypeError("must be finite and greater than 1")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return result


def validate_instance(
    disutilities: np.ndarray, budgets: np.ndarray, rho: float
) -> None:
    if disutilities.ndim != 2 or 0 in disutilities.shape:
        raise ValueError("disutilities must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(disutilities)) or np.any(disutilities <= 0):
        raise ValueError("disutilities must contain only finite, positive values")
    if budgets.shape != (disutilities.shape[0],):
        raise ValueError("budgets must contain one entry per agent")
    if not np.all(np.isfinite(budgets)) or np.any(budgets < 0) or budgets.sum() <= 0:
        raise ValueError("budgets must be finite, nonnegative, and have a positive sum")
    if not np.isfinite(rho) or rho <= 1:
        raise ValueError("rho must be finite and greater than 1")


def compute_update(
    method: str, prices: np.ndarray, excess_supply: np.ndarray
) -> np.ndarray:
    """Center the original additive, multiplicative, or quadratic update."""
    if method == "additive":
        return excess_supply - excess_supply.mean()
    if method == "multiplicative":
        update = prices * excess_supply
    elif method == "quadratic":
        update = (prices**2) * excess_supply
    else:
        raise ValueError(f"Unknown method: {method}")
    return update - update.mean()


def run_instance(
    disutilities: np.ndarray,
    budgets: np.ndarray,
    method: str,
    eta: float,
    max_steps: int,
    rho: float,
) -> RunResult:
    """Run one candidate, preserving the original failure and stopping rules."""
    validate_instance(disutilities, budgets, rho)
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    prices = np.full(
        disutilities.shape[1], budgets.sum() / disutilities.shape[1], dtype=float
    )
    hit_iterations = dict.fromkeys(EPSILONS)
    for iteration in range(max_steps):
        excess_supply = compute_excess_supply(prices, disutilities, budgets, rho)
        prices = prices + eta * compute_update(method, prices, excess_supply)
        if np.any(prices < 0):
            return False, "negative prices"
        if not np.all(np.isfinite(prices)):
            return False, "blowup - nonfinite prices"

        norm = np.linalg.norm(excess_supply)
        for epsilon in EPSILONS:
            if norm <= epsilon and hit_iterations[epsilon] is None:
                hit_iterations[epsilon] = iteration
        if all(value is not None for value in hit_iterations.values()):
            break
    return True, hit_iterations


def search_stepsize(
    evaluate: Callable[[float], RunResult],
    max_steps: int,
    *,
    verbose: bool = True,
    max_rounds: int = 1000,
) -> tuple[float, dict]:
    """Apply the original nine-candidate shrinking-grid search.

    The historical function name says "largest", but the selected candidate is
    the fastest to reach 0.001 in the first grid with no failed runs. Reaching
    max_steps without convergence is still a successful run, as in the originals.
    """
    if max_steps < 1 or max_rounds < 1:
        raise ValueError("max_steps and max_rounds must be positive")
    upper_bound = 10
    for _ in range(max_rounds):
        candidates = [upper_bound * (index + 1) / 10 for index in range(9)]
        if candidates[0] <= 0:
            raise RuntimeError("Step-size search reached floating-point underflow")
        successes = []
        scores = []
        for eta in candidates:
            ok, data = evaluate(eta)
            successes.append(ok)
            if ok:
                hit = data[0.001]
                scores.append(hit if hit is not None else max_steps + 1)
            else:
                scores.append(2 * max_steps + 1)
        if verbose:
            print(candidates, scores)
        best_eta = candidates[np.argsort(scores)[0]]
        if all(successes):
            ok, data = evaluate(best_eta)
            if not ok:
                raise RuntimeError(f"Selected step size failed on repeat: {data}")
            if verbose:
                print(data)
            return best_eta, data
        upper_bound = min(eta for eta, ok in zip(candidates, successes) if not ok)
    raise RuntimeError(
        f"Step-size search failed to find a stable grid after {max_rounds} rounds"
    )


def append_result(output_dir: Path, filename: str, row: dict) -> None:
    """Append one experiment result without overwriting earlier runs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    frame = pd.DataFrame([row])
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    if not write_header:
        with output_path.open(newline="", encoding="utf-8") as source:
            columns = next(csv.reader(source))
        if columns != [str(column) for column in frame.columns]:
            raise ValueError(
                f"Cannot append results with different columns to {output_path}; "
                "choose a fresh --output-dir"
            )
    print(frame)
    frame.to_csv(output_path, index=False, mode="a", header=write_header)
