from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import compute_excess_supply


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep stepsizes for additive and multiplicative tatonnement."
    )
    parser.add_argument("--num-instances", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=30000)
    parser.add_argument("--num-etas", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[1.2, 1.5, 2.0, 3.0, 5.0, 10.0],
    )
    return parser.parse_args()


def _simulate_until_epsilon(
    method: str,
    prices_init: np.ndarray,
    disutility_weights: np.ndarray,
    earning_requirements: np.ndarray,
    rho: float,
    eta: float,
    epsilon: float,
    num_steps: int,
) -> tuple[str, int | None, float | None]:
    prices = prices_init.astype(float).copy()

    for t in range(num_steps):
        try:
            z = compute_excess_supply(prices, disutility_weights, earning_requirements, rho)
            if not np.all(np.isfinite(z)):
                return "blowup - nonfinite excess supply", None, None
            norm = float(np.linalg.norm(z))
            if not np.isfinite(norm):
                return "blowup - nonfinite norm", None, None

            if norm <= epsilon:
                return "converged", t, norm

            if method == "additive":
                update = z - z.mean()
            elif method == "multiplicative":
                u = prices * z
                update = u - u.mean()  # for correcting any numerical error
            elif method == "quadratic":
                u = (prices ** 2) * z
                update = u - u.mean()
            else:
                raise ValueError(f"Unknown method: {method}")

            next_prices = prices + eta * update

            # If an update would make one of prices nonpositive, mark it separately.
            if np.any(next_prices < 0):
                return "negative prices", None, None

            prices = next_prices
            
            if not np.all(np.isfinite(prices)):
                return "blowup - nonfinite prices", None, None
        except Exception:
            return "blowup - exception", None, None

    # Budget exhausted without hitting epsilon.
    try:
        z_end = compute_excess_supply(prices, disutility_weights, earning_requirements, rho)
        end_norm = float(np.linalg.norm(z_end))
    except Exception:
        return "blowup - exception", None, None
    return "budget_exhausted", None, end_norm


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parent

    n_agents = 100
    n_resources = 500
    rhos = args.rhos
    epsilon = 0.001
    num_steps = args.num_steps
    num_instances = args.num_instances
    seed = args.seed

    # Same eta grid for all methods.
    eta_values = np.logspace(-5, 0, args.num_etas)
    methods = ["additive", "multiplicative"]
    output_stem = f"tatonnement_synthetic_{n_agents}x{n_resources}_stepsize"
    if args.output_tag:
        output_stem = f"{output_stem}_{args.output_tag}"

    rng = np.random.default_rng(seed)
    instances = []
    for instance_idx in range(num_instances):
        D = rng.lognormal(mean=0.0, sigma=1.0, size=(n_agents, n_resources))
        B = rng.lognormal(mean=0.0, sigma=0.5, size=n_agents)
        p0 = np.full(n_resources, B.sum() / n_resources, dtype=float)
        instances.append((instance_idx, D, B, p0))

    rows: list[dict[str, float | int | str | None]] = []

    print(
        f"Synthetic instances: count={num_instances}, n={n_agents}, m={n_resources}, "
        f"rhos={rhos}, epsilon={epsilon}, budget={num_steps}, etas={len(eta_values)}"
    )

    for rho in rhos:
        print(f"\n##### rho={rho} #####")
        for method in methods:
            print(f"\n=== {method} ===")
            for eta in eta_values:
                converged_iters: list[int] = []
                status_counts: dict[str, int] = {}

                for instance_idx, D, B, p0 in instances:
                    status, iters, end_norm = _simulate_until_epsilon(
                        method=method,
                        prices_init=p0,
                        disutility_weights=D,
                        earning_requirements=B,
                        rho=float(rho),
                        eta=float(eta),
                        epsilon=epsilon,
                        num_steps=num_steps,
                    )
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if status == "converged" and iters is not None:
                        converged_iters.append(iters)

                    rows.append(
                        {
                            "rho": float(rho),
                            "instance_idx": instance_idx,
                            "method": method,
                            "eta": float(eta),
                            "epsilon": epsilon,
                            "iteration_budget": num_steps,
                            "status": status,
                            "iterations_to_epsilon": iters,
                            "final_norm": end_norm,
                        }
                    )

                if converged_iters:
                    avg_iters = float(np.mean(converged_iters))
                    print(
                        f"  eta={eta:.6f}: {len(converged_iters)}/{num_instances} converged "
                        f"(avg iterations={avg_iters:.1f}, statuses={status_counts})"
                    )
                else:
                    print(
                        f"  eta={eta:.6f}: 0/{num_instances} converged "
                        f"(statuses={status_counts})"
                    )

    df = pd.DataFrame(rows)

    csv_path = root / f"{output_stem}_scatter_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved data to: {csv_path}")

    df["iterations_with_failures_as_budget"] = df["iterations_to_epsilon"].where(
        df["status"] == "converged",
        num_steps + 1,
    )
    summary = (
        df.groupby(["rho", "method", "eta", "epsilon", "iteration_budget"], as_index=False)
        .agg(
            converged_count=("status", lambda s: int((s == "converged").sum())),
            avg_iterations_to_epsilon=("iterations_to_epsilon", "mean"),
            median_iterations_to_epsilon=("iterations_to_epsilon", "median"),
            avg_iterations_with_failures_as_budget=(
                "iterations_with_failures_as_budget",
                "mean",
            ),
            avg_final_norm=("final_norm", "mean"),
        )
    )
    summary["num_instances"] = num_instances
    summary["convergence_rate"] = summary["converged_count"] / num_instances

    summary_csv_path = (
        root / f"{output_stem}_scatter_summary.csv"
    )
    summary.to_csv(summary_csv_path, index=False)
    print(f"Saved summary to: {summary_csv_path}")

    ncols = 3
    nrows = int(np.ceil(len(rhos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows), squeeze=False)
    marker_by_method = {
        "additive": "o",
        "multiplicative": "s",
    }
    color_by_method = {
        "additive": "C0",
        "multiplicative": "C1",
    }

    for ax, rho in zip(axes.ravel(), rhos):
        for method in methods:
            sub = summary[
                (summary["rho"] == float(rho))
                & (summary["method"] == method)
                & (summary["converged_count"] == summary["num_instances"])
            ].sort_values("eta")
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["eta"],
                sub["avg_iterations_to_epsilon"],
                label=method,
                marker=marker_by_method[method],
                color=color_by_method[method],
                s=36,
                alpha=0.9,
            )
            ax.plot(
                sub["eta"],
                sub["avg_iterations_to_epsilon"],
                color=color_by_method[method],
                linewidth=1.1,
                alpha=0.8,
            )

        ax.set_xscale("log")
        ax.set_title(f"rho={rho:g}")
        ax.set_xlabel("stepsize eta")
        ax.set_ylabel("avg iterations")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels)

    for ax in axes.ravel()[len(rhos):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Synthetic {n_agents}x{n_resources}: Additive vs Multiplicative "
        f"({num_instances} instances, epsilon={epsilon:g})"
    )
    fig.tight_layout()

    plot_path = root / f"{output_stem}_vs_iterations_scatter.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
