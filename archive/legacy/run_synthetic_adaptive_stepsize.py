from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import compute_excess_supply


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run additive, multiplicative, and quadratic tatonnement with a "
            "stepsize that backs off whenever the proposed update makes a "
            "price negative."
        )
    )
    parser.add_argument("--n-agents", type=int, default=100)
    parser.add_argument("--n-resources", type=int, default=500)
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=30000)
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.1, 0.05, 0.01, 0.005, 0.001],
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Deprecated alias for running a single epsilon threshold.",
    )
    parser.add_argument("--initial-eta", type=float, default=10.0)
    parser.add_argument("--backoff-factor", type=float, default=0.8)
    parser.add_argument("--min-eta", type=float, default=1e-18)
    parser.add_argument("--price-floor", type=float, default=1e-18)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[1.2, 1.5, 2.0, 3.0, 5.0, 10.0],
    )
    return parser.parse_args()


def _compute_update(method: str, prices: np.ndarray, excess_supply: np.ndarray) -> np.ndarray:
    if method == "additive":
        return excess_supply - excess_supply.mean()
    if method == "multiplicative":
        u = prices * excess_supply
        return u - u.mean()
    if method == "quadratic":
        u = (prices ** 2) * excess_supply
        return u - u.mean()
    raise ValueError(f"Unknown method: {method}")


def _max_positive_eta(prices: np.ndarray, update: np.ndarray) -> float:
    decreasing = update < 0
    if not np.any(decreasing):
        return np.inf
    return float(np.min(-prices[decreasing] / update[decreasing]))


def _mean_finite(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def _median_finite(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.median())


def _simulate_adaptive_until_epsilon(
    method: str,
    prices_init: np.ndarray,
    disutility_weights: np.ndarray,
    earning_requirements: np.ndarray,
    rho: float,
    initial_eta: float,
    backoff_factor: float,
    min_eta: float,
    price_floor: float,
    epsilons: list[float],
    num_steps: int,
) -> dict[str, float | int | str | None]:
    prices = prices_init.astype(float).copy()
    eta = float(initial_eta)
    sorted_epsilons = sorted(float(eps) for eps in epsilons)
    target_epsilon = sorted_epsilons[0]
    hitting_iterations: dict[float, int | None] = {eps: None for eps in sorted_epsilons}

    accepted_etas: list[float] = []
    admissible_bounds: list[float] = []
    total_backoffs = 0
    max_backoffs_single_step = 0
    final_norm: float | None = None

    for t in range(num_steps):
        try:
            z = compute_excess_supply(prices, disutility_weights, earning_requirements, rho)
            if not np.all(np.isfinite(z)):
                return {
                    "status": "blowup - nonfinite excess supply",
                    "hitting_iterations": hitting_iterations,
                    "final_norm": None,
                    "final_eta": eta,
                    "min_accepted_eta": min(accepted_etas) if accepted_etas else None,
                    "max_accepted_eta": max(accepted_etas) if accepted_etas else None,
                    "mean_accepted_eta": float(np.mean(accepted_etas)) if accepted_etas else None,
                    "mean_admissible_eta_bound": (
                        float(np.mean(admissible_bounds)) if admissible_bounds else None
                    ),
                    "total_backoffs": total_backoffs,
                    "max_backoffs_single_step": max_backoffs_single_step,
                }

            norm = float(np.linalg.norm(z))
            if not np.isfinite(norm):
                return {
                    "status": "blowup - nonfinite norm",
                    "hitting_iterations": hitting_iterations,
                    "final_norm": None,
                    "final_eta": eta,
                    "min_accepted_eta": min(accepted_etas) if accepted_etas else None,
                    "max_accepted_eta": max(accepted_etas) if accepted_etas else None,
                    "mean_accepted_eta": float(np.mean(accepted_etas)) if accepted_etas else None,
                    "mean_admissible_eta_bound": (
                        float(np.mean(admissible_bounds)) if admissible_bounds else None
                    ),
                    "total_backoffs": total_backoffs,
                    "max_backoffs_single_step": max_backoffs_single_step,
                }
            final_norm = norm

            for eps in sorted_epsilons:
                if hitting_iterations[eps] is None and norm <= eps:
                    hitting_iterations[eps] = t

            if norm <= target_epsilon:
                return {
                    "status": "converged",
                    "hitting_iterations": hitting_iterations,
                    "final_norm": norm,
                    "final_eta": eta,
                    "min_accepted_eta": min(accepted_etas) if accepted_etas else None,
                    "max_accepted_eta": max(accepted_etas) if accepted_etas else None,
                    "mean_accepted_eta": float(np.mean(accepted_etas)) if accepted_etas else None,
                    "mean_admissible_eta_bound": (
                        float(np.mean(admissible_bounds)) if admissible_bounds else None
                    ),
                    "total_backoffs": total_backoffs,
                    "max_backoffs_single_step": max_backoffs_single_step,
                }

            update = _compute_update(method, prices, z)
            admissible_bounds.append(_max_positive_eta(prices, update))

            backoffs_this_step = 0
            while True:
                next_prices = prices + eta * update
                if np.all(np.isfinite(next_prices)) and np.all(next_prices >= price_floor):
                    break
                eta *= backoff_factor
                backoffs_this_step += 1
                if eta < min_eta:
                    return {
                        "status": "eta below minimum",
                        "hitting_iterations": hitting_iterations,
                        "final_norm": norm,
                        "final_eta": eta,
                        "min_accepted_eta": min(accepted_etas) if accepted_etas else None,
                        "max_accepted_eta": max(accepted_etas) if accepted_etas else None,
                        "mean_accepted_eta": (
                            float(np.mean(accepted_etas)) if accepted_etas else None
                        ),
                        "mean_admissible_eta_bound": (
                            float(np.mean(admissible_bounds)) if admissible_bounds else None
                        ),
                        "total_backoffs": total_backoffs + backoffs_this_step,
                        "max_backoffs_single_step": max(
                            max_backoffs_single_step,
                            backoffs_this_step,
                        ),
                    }

            total_backoffs += backoffs_this_step
            max_backoffs_single_step = max(max_backoffs_single_step, backoffs_this_step)
            accepted_etas.append(eta)
            prices = next_prices
        except Exception:
            return {
                "status": "blowup - exception",
                "hitting_iterations": hitting_iterations,
                "final_norm": final_norm,
                "final_eta": eta,
                "min_accepted_eta": min(accepted_etas) if accepted_etas else None,
                "max_accepted_eta": max(accepted_etas) if accepted_etas else None,
                "mean_accepted_eta": float(np.mean(accepted_etas)) if accepted_etas else None,
                "mean_admissible_eta_bound": (
                    float(np.mean(admissible_bounds)) if admissible_bounds else None
                ),
                "total_backoffs": total_backoffs,
                "max_backoffs_single_step": max_backoffs_single_step,
            }

    return {
        "status": "budget_exhausted",
        "hitting_iterations": hitting_iterations,
        "final_norm": final_norm,
        "final_eta": eta,
        "min_accepted_eta": min(accepted_etas) if accepted_etas else None,
        "max_accepted_eta": max(accepted_etas) if accepted_etas else None,
        "mean_accepted_eta": float(np.mean(accepted_etas)) if accepted_etas else None,
        "mean_admissible_eta_bound": (
            float(np.mean(admissible_bounds)) if admissible_bounds else None
        ),
        "total_backoffs": total_backoffs,
        "max_backoffs_single_step": max_backoffs_single_step,
    }


def _make_summary_plot(
    summary: pd.DataFrame,
    methods: list[str],
    rhos: list[float],
    output_path: Path,
    num_instances: int,
) -> None:
    ncols = 3
    nrows = int(np.ceil(len(rhos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows), squeeze=False)
    marker_by_method = {
        "additive": "o",
        "multiplicative": "s",
        "quadratic": "^",
    }

    for ax, rho in zip(axes.ravel(), rhos):
        for method in methods:
            sub = summary[
                (summary["rho"] == float(rho)) & (summary["method"] == method)
            ].sort_values("epsilon", ascending=False)
            if len(sub) == 0:
                continue
            ax.plot(
                sub["epsilon"],
                sub["avg_iterations_with_failures_as_budget"],
                marker=marker_by_method[method],
                linewidth=1.4,
                label=method,
            )

        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("epsilon")
        ax.set_ylabel("avg total iterations")
        ax.set_title(f"rho={rho:g}")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)

    for ax in axes.ravel()[len(rhos):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Adaptive stepsize tatonnement: total iterations vs epsilon "
        f"({num_instances} instances)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    if not (0 < args.backoff_factor < 1):
        raise ValueError("--backoff-factor must be between 0 and 1")
    if args.initial_eta <= 0:
        raise ValueError("--initial-eta must be positive")
    if args.min_eta <= 0:
        raise ValueError("--min-eta must be positive")
    if args.price_floor < 0:
        raise ValueError("--price-floor must be nonnegative")
    epsilons = [float(args.epsilon)] if args.epsilon is not None else [float(eps) for eps in args.epsilons]
    if any(eps <= 0 for eps in epsilons):
        raise ValueError("epsilon thresholds must be positive")
    epsilons = sorted(set(epsilons), reverse=True)

    root = Path(__file__).resolve().parent
    methods = ["additive", "multiplicative", "quadratic"]
    rhos = [float(rho) for rho in args.rhos]

    output_stem = f"tatonnement_synthetic_{args.n_agents}x{args.n_resources}_adaptive_stepsize"
    if args.output_tag:
        output_stem = f"{output_stem}_{args.output_tag}"

    rng = np.random.default_rng(args.seed)
    instances = []
    for instance_idx in range(args.num_instances):
        D = rng.lognormal(mean=0.0, sigma=1.0, size=(args.n_agents, args.n_resources))
        B = rng.lognormal(mean=0.0, sigma=0.5, size=args.n_agents)
        p0 = np.full(args.n_resources, B.sum() / args.n_resources, dtype=float)
        instances.append((instance_idx, D, B, p0))

    rows: list[dict[str, float | int | str | None]] = []
    print(
        f"Adaptive stepsize synthetic experiment: n={args.n_agents}, "
        f"m={args.n_resources}, instances={args.num_instances}, rhos={rhos}, "
        f"initial_eta={args.initial_eta}, backoff={args.backoff_factor}, "
        f"epsilons={epsilons}, budget={args.num_steps}"
    )

    for rho in rhos:
        print(f"\n##### rho={rho:g} #####")
        for method in methods:
            print(f"\n=== {method} ===")
            status_counts: dict[str, int] = {}
            for instance_idx, D, B, p0 in instances:
                result = _simulate_adaptive_until_epsilon(
                    method=method,
                    prices_init=p0,
                    disutility_weights=D,
                    earning_requirements=B,
                    rho=rho,
                    initial_eta=args.initial_eta,
                    backoff_factor=args.backoff_factor,
                    min_eta=args.min_eta,
                    price_floor=args.price_floor,
                    epsilons=epsilons,
                    num_steps=args.num_steps,
                )
                status = str(result["status"])
                status_counts[status] = status_counts.get(status, 0) + 1
                hitting_iterations = result.pop("hitting_iterations")
                for eps in epsilons:
                    hit_iter = hitting_iterations[float(eps)]
                    rows.append(
                        {
                            "rho": rho,
                            "instance_idx": instance_idx,
                            "method": method,
                            "initial_eta": args.initial_eta,
                            "backoff_factor": args.backoff_factor,
                            "epsilon": eps,
                            "target_epsilon": min(epsilons),
                            "iteration_budget": args.num_steps,
                            **result,
                            "status": "converged" if hit_iter is not None else status,
                            "terminal_status": status,
                            "iterations_to_epsilon": hit_iter,
                        }
                    )
                print(
                    f"  instance={instance_idx:2d}: {status}, "
                    f"iters_to_min_epsilon={hitting_iterations[min(epsilons)]}, "
                    f"final_eta={result['final_eta']:.6g}, "
                    f"backoffs={result['total_backoffs']}"
                )
            print(f"  statuses={status_counts}")

    df = pd.DataFrame(rows)
    data_path = root / f"{output_stem}_data.csv"
    df.to_csv(data_path, index=False)
    print(f"\nSaved data to: {data_path}")

    df["iterations_with_failures_as_budget"] = df["iterations_to_epsilon"].where(
        df["status"] == "converged",
        args.num_steps + 1,
    )
    summary = (
        df.groupby(["rho", "method", "epsilon"], as_index=False)
        .agg(
            converged_count=("status", lambda s: int((s == "converged").sum())),
            avg_iterations_to_epsilon=("iterations_to_epsilon", _mean_finite),
            median_iterations_to_epsilon=("iterations_to_epsilon", _median_finite),
            avg_iterations_with_failures_as_budget=(
                "iterations_with_failures_as_budget",
                _mean_finite,
            ),
            avg_final_norm=("final_norm", _mean_finite),
            avg_final_eta=("final_eta", _mean_finite),
            median_final_eta=("final_eta", _median_finite),
            avg_min_accepted_eta=("min_accepted_eta", _mean_finite),
            avg_max_accepted_eta=("max_accepted_eta", _mean_finite),
            avg_mean_accepted_eta=("mean_accepted_eta", _mean_finite),
            avg_mean_admissible_eta_bound=("mean_admissible_eta_bound", _mean_finite),
            avg_total_backoffs=("total_backoffs", _mean_finite),
            max_backoffs_single_step=("max_backoffs_single_step", "max"),
        )
    )
    summary["num_instances"] = args.num_instances
    summary["convergence_rate"] = summary["converged_count"] / args.num_instances

    summary_path = root / f"{output_stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

    plot_path = root / f"{output_stem}_summary.png"
    _make_summary_plot(summary, methods, rhos, plot_path, args.num_instances)
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
