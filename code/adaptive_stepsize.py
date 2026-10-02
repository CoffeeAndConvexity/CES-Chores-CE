"""Compare adaptive-step-size updates on synthetic disutility matrices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiment_utils import (
    DEFAULT_OUTPUT_DIR,
    METHODS,
    RunResult,
    append_result,
    ces_rho,
    compute_update as _compute_update,
    positive_int,
    run_instance,
    search_stepsize,
)

DISTRIBUTIONS = (
    "lognormal",
    "uniform",
    "integer_uniform",
    "exponential",
    "truncated_normal",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for a stable, fast step size on synthetic instances."
    )
    parser.add_argument("--n-agents", type=positive_int, default=100)
    parser.add_argument("--n-resources", type=positive_int, default=200)
    parser.add_argument("--rho", type=ces_rho, default=2.0)
    parser.add_argument("--num-instances", type=positive_int, default=5)
    parser.add_argument("--max-steps", type=positive_int, default=10000)
    parser.add_argument(
        "--value-distribution", choices=DISTRIBUTIONS, default="lognormal"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def generate_instance(
    n_agents: int,
    n_resources: int,
    value_distribution: str,
    instance_seed: int,
) -> np.ndarray:
    """Use the original MT19937 draws without changing NumPy's global RNG."""
    if n_agents < 1 or n_resources < 1:
        raise ValueError("n_agents and n_resources must be positive")
    rng = np.random.RandomState(instance_seed)
    shape = (n_agents, n_resources)
    if value_distribution == "lognormal":
        return rng.lognormal(mean=0.0, sigma=1.0, size=shape)
    if value_distribution == "uniform":
        return rng.uniform(low=0.0, high=1.0, size=shape)
    if value_distribution == "integer_uniform":
        return rng.randint(low=1, high=20, size=shape)
    if value_distribution == "exponential":
        return rng.exponential(scale=1.0, size=shape)
    if value_distribution == "truncated_normal":
        return np.clip(
            rng.normal(loc=0.5, scale=0.2, size=shape), a_min=0.01, a_max=None
        )
    raise ValueError(f"Unknown value distribution: {value_distribution}")


def run(
    method: str,
    eta: float,
    max_steps: int,
    instance_seed: int,
    *,
    n_agents: int = 100,
    n_resources: int = 200,
    rho: float = 2.0,
    value_distribution: str = "lognormal",
) -> RunResult:
    disutilities = generate_instance(
        n_agents, n_resources, value_distribution, instance_seed
    )
    ok, data = run_instance(
        disutilities, np.ones(n_agents), method, eta, max_steps, rho
    )
    return (True, {"instance": instance_seed, **data}) if ok else (False, data)


def search_largest_stepsize(
    method: str = "multiplicative",
    max_steps: int = 10000,
    instance_seed: int = 0,
    *,
    n_agents: int = 100,
    n_resources: int = 200,
    rho: float = 2.0,
    value_distribution: str = "lognormal",
) -> tuple[float, dict]:
    # All candidates use the same matrix, as they did when each run reset the seed.
    disutilities = generate_instance(
        n_agents, n_resources, value_distribution, instance_seed
    )
    budgets = np.ones(n_agents)

    def evaluate(eta: float) -> RunResult:
        ok, data = run_instance(disutilities, budgets, method, eta, max_steps, rho)
        return (True, {"instance": instance_seed, **data}) if ok else (False, data)

    return search_stepsize(evaluate, max_steps)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    for method in METHODS:
        for instance_seed in range(args.num_instances):
            best_eta, data = search_largest_stepsize(
                method,
                args.max_steps,
                instance_seed,
                n_agents=args.n_agents,
                n_resources=args.n_resources,
                rho=args.rho,
                value_distribution=args.value_distribution,
            )
            append_result(
                args.output_dir,
                f"batch_adaptive_stepsize_{method}_{args.n_agents}_{args.n_resources}_{args.value_distribution}.csv",
                {
                    "method": method,
                    "n_agents": args.n_agents,
                    "n_resources": args.n_resources,
                    "rho": args.rho,
                    "instance": instance_seed,
                    "eta": best_eta,
                    **data,
                },
            )


if __name__ == "__main__":
    main()
