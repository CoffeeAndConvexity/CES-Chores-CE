"""Compare adaptive-step-size updates on sampled paper-bidding data."""

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
    nonnegative_float,
    positive_int,
    run_instance,
    search_stepsize,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search for a stable, fast step size on bidding instances."
    )
    parser.add_argument("--rho", type=ces_rho, default=2.0)
    parser.add_argument("--num-instances", type=positive_int, default=5)
    parser.add_argument("--max-steps", type=positive_int, default=10000)
    parser.add_argument("--noise-level", type=nonnegative_float, default=1.0)
    parser.add_argument(
        "--N", type=positive_int, default=100, help="Number of sampled reviewers."
    )
    parser.add_argument(
        "--M", type=positive_int, default=200, help="Number of sampled papers."
    )
    parser.add_argument(
        "--bidding-data", type=Path, default=REPO_ROOT / "data" / "bidding-data.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if not args.bidding_data.is_file():
        parser.error(f"bidding data does not exist: {args.bidding_data}")
    return args


def run(
    D_instance: np.ndarray,
    method: str,
    eta: float,
    max_steps: int,
    instance_seed: int,
    *,
    rho: float = 2.0,
) -> RunResult:
    ok, data = run_instance(
        D_instance, np.ones(D_instance.shape[0]), method, eta, max_steps, rho
    )
    return (True, {"instance": instance_seed, **data}) if ok else (False, data)


def search_largest_stepsize(
    D_instance: np.ndarray,
    method: str = "multiplicative",
    max_steps: int = 10000,
    instance_seed: int = 0,
    *,
    rho: float = 2.0,
) -> tuple[float, dict]:
    return search_stepsize(
        lambda eta: run(D_instance, method, eta, max_steps, instance_seed, rho=rho),
        max_steps,
    )


def load_bidding_data(path: Path, n_sampled_papers: int) -> np.ndarray:
    """Encode bids with the original preference and conflict weights."""
    frame = pd.read_csv(path)
    columns = ["Bidder", "Submission", "Bid"]
    if any(column not in frame for column in columns) or frame.empty:
        raise ValueError(
            "Bidding data must contain nonempty Bidder, Submission, and Bid columns"
        )
    if frame[columns].isna().any().any():
        raise ValueError(
            "Bidding data cannot contain missing bidders, submissions, or bids"
        )
    submissions = pd.to_numeric(frame["Submission"], errors="coerce")
    if (
        not np.isfinite(submissions).all()
        or (submissions < 1).any()
        or (submissions % 1 != 0).any()
    ):
        raise ValueError("Submission identifiers must be positive integers")
    preference_values = {
        "yes": 1,
        "maybe": 3,
        "no response": 5,
        "no": 7,
        "conflict": 7 * n_sampled_papers + 1,
    }
    unknown = set(frame["Bid"]) - preference_values.keys()
    if unknown:
        raise ValueError(f"Unknown bid values: {sorted(unknown)}")
    bidder_indices = {
        bidder: index for index, bidder in enumerate(dict.fromkeys(frame["Bidder"]))
    }
    disutilities = np.full((len(bidder_indices), int(submissions.max())), 5.0)
    for bidder, submission, bid in frame[columns].itertuples(index=False, name=None):
        disutilities[bidder_indices[bidder], int(submission) - 1] = preference_values[
            bid
        ]
    return disutilities


def distance_matrix_among_papers(disutilities: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances, retaining the original summation order."""
    n_papers = disutilities.shape[1]
    papers = disutilities.T
    distances = np.zeros((n_papers, n_papers))
    for first in range(n_papers):
        for second in range(n_papers):
            distances[first, second] = sum((papers[first] - papers[second]) ** 2)
    return distances


def sample_bidding_instance(
    disutilities: np.ndarray,
    distances: np.ndarray,
    n_agents: int,
    n_resources: int,
    instance_seed: int,
    noise_level: float,
) -> np.ndarray:
    """Preserve seeded rejection sampling and separately reseeded Gaussian noise.

    Check that at least one anchor paper can pass the historical acceptance rule
    before rejection sampling, so impossible requests cannot loop forever.
    """
    if (
        not 1 <= n_agents <= disutilities.shape[0]
        or not 1 <= n_resources <= disutilities.shape[1]
    ):
        raise ValueError(
            f"Requested sample ({n_agents}, {n_resources}) exceeds bidding matrix {disutilities.shape}"
        )
    if not np.isfinite(noise_level) or noise_level < 0:
        raise ValueError("noise_level must be finite and nonnegative")

    def sample_for_anchor(anchor: int) -> np.ndarray:
        nearest = np.argsort(distances[anchor])[:n_resources]
        response_counts = 1 + np.sum(disutilities[:, nearest] < 5 - 1e-3, axis=1)
        reviewers = np.flip(np.argsort(response_counts))[:n_agents]
        return disutilities[np.ix_(reviewers, nearest)]

    def accepted(sample: np.ndarray) -> bool:
        return bool(np.max(np.min(sample, axis=1)) < 6)

    if not any(
        accepted(sample_for_anchor(anchor)) for anchor in range(disutilities.shape[1])
    ):
        raise ValueError(
            "No anchor paper produces an acceptable bidding instance at the requested dimensions"
        )
    rng = np.random.RandomState(instance_seed)
    while True:
        sampled = sample_for_anchor(rng.randint(disutilities.shape[1]))
        if accepted(sampled):
            break
    if noise_level > 0:
        noise_rng = np.random.RandomState(instance_seed)
        noise = noise_level * noise_rng.normal(size=(n_agents, n_resources))
        sampled = np.maximum(sampled + noise, 1)
    return sampled


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    print(f"Loading bidding data from {args.bidding_data}")
    disutilities = load_bidding_data(args.bidding_data, args.M)
    if args.N > disutilities.shape[0] or args.M > disutilities.shape[1]:
        raise ValueError(
            f"Requested sample ({args.N}, {args.M}) exceeds bidding matrix {disutilities.shape}"
        )
    distances = distance_matrix_among_papers(disutilities)
    for instance_seed in range(args.num_instances):
        sampled = sample_bidding_instance(
            disutilities,
            distances,
            args.N,
            args.M,
            instance_seed,
            args.noise_level,
        )
        print(f"Testing instance {instance_seed} with shape {sampled.shape}")
        for method in METHODS:
            best_eta, data = search_largest_stepsize(
                sampled,
                method,
                args.max_steps,
                instance_seed,
                rho=args.rho,
            )
            append_result(
                args.output_dir,
                f"batch_adaptive_stepsize_{method}_bidding.csv",
                {
                    "method": method,
                    "n_agents": args.N,
                    "n_resources": args.M,
                    "rho": args.rho,
                    "instance": instance_seed,
                    "eta": best_eta,
                    **data,
                    "noise_level": args.noise_level,
                },
            )


if __name__ == "__main__":
    main()
