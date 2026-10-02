from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import (
    compute_excess_supply,
    run_multiplicative_tatonnement,
    run_tatonnement,
)


def _load_disutility_matrix(matrix_path: Path) -> np.ndarray:
    df = pd.read_csv(matrix_path)
    if "agent_id" in df.columns:
        df = df.drop(columns=["agent_id"])
    D = df.to_numpy(dtype=float)
    if D.ndim != 2:
        raise ValueError(f"Expected 2D matrix in {matrix_path}")
    if np.any(D <= 0):
        raise ValueError(f"Found non-positive disutility entries in {matrix_path}")
    return D


def _first_hitting_iteration(norms: np.ndarray, epsilon: float) -> int:
    hit_idx = np.where(norms <= epsilon)[0]
    if hit_idx.size == 0:
        return norms.size + 1
    return int(hit_idx[0])


def main() -> None:
    root = Path(__file__).resolve().parent
    matrix_dir = root / "Spliddit_Data" / "distribute_tasks_valuation_matrices"
    index_path = matrix_dir / "_index.csv"

    index_df = pd.read_csv(index_path)
    index_df["market_size"] = index_df["num_agents"] * index_df["num_resources"]
    largest_20 = index_df.nlargest(20, "market_size").copy()
    chosen_ids = sorted(largest_20["instance_id"].tolist())

    rho = 2.0
    eta_values = [0.001, 0.002, 0.005]
    num_steps = 100000
    epsilons = [0.1, 0.05, 0.01, 0.005, 0.001]

    print(f"Running {len(chosen_ids)} largest instances: {chosen_ids}")
    print(f"Parameters: rho={rho}, num_steps={num_steps}")
    print(f"Testing eta values: {eta_values}")

    # Dictionary: {(variant, eta, epsilon) -> avg_iterations}
    results = {}

    for variant in ["additive", "multiplicative"]:
        print(f"\n{'='*60}")
        print(f"Variant: {variant.upper()}")
        print(f"{'='*60}")

        for eta in eta_values:
            print(f"\neta={eta}:")
            rows_this_config = []

            for instance_id in chosen_ids:
                row = index_df[index_df["instance_id"] == instance_id].iloc[0]
                matrix_file = str(row["matrix_file"])

                D = _load_disutility_matrix(matrix_dir / matrix_file)
                n, m = D.shape
                B = np.ones(n, dtype=float)
                p0 = np.full(m, B.sum() / m, dtype=float)

                if variant == "additive":
                    prices, excess_supplies = run_tatonnement(
                        prices_init=p0,
                        disutility_weights=D,
                        earning_requirements=B,
                        rho=rho,
                        eta=eta,
                        num_steps=num_steps,
                    )
                else:  # multiplicative
                    prices, excess_supplies = run_multiplicative_tatonnement(
                        prices_init=p0,
                        disutility_weights=D,
                        earning_requirements=B,
                        rho=rho,
                        eta=eta,
                        num_steps=num_steps,
                    )

                norms = np.linalg.norm(excess_supplies, axis=1)
                print(
                    f"  instance={instance_id:5d}  ||z_0||={norms[0]:.6f}  ||z_T||={norms[-1]:.6f}"
                )

                for eps in epsilons:
                    t_hit = _first_hitting_iteration(norms, eps)
                    rows_this_config.append(
                        {
                            "epsilon": eps,
                            "iterations_to_epsilon": t_hit,
                        }
                    )

            # Compute average for this (variant, eta)
            df_config = pd.DataFrame(rows_this_config)
            for eps in epsilons:
                avg_iters = df_config[df_config["epsilon"] == eps]["iterations_to_epsilon"].mean()
                results[(variant, eta, eps)] = avg_iters

    # Generate comparison plot
    plt.figure(figsize=(14, 8))

    eta_colors = {0.001: "C0", 0.002: "C1", 0.005: "C2"}
    eta_markers = {0.001: "o", 0.002: "s", 0.005: "^"}
    
    for variant, linestyle in [("additive", "-"), ("multiplicative", "--")]:
        for eta in eta_values:
            inv_eps = 1.0 / np.array(epsilons)
            avg_iters = np.array([results.get((variant, eta, eps), num_steps + 1) for eps in epsilons])

            label = f"{variant} eta={eta}"
            plt.plot(
                inv_eps,
                avg_iters,
                marker=eta_markers[eta],
                linestyle=linestyle,
                linewidth=2.0,
                markersize=7,
                color=eta_colors[eta],
                label=label,
            )

    plt.xscale("log")
    plt.xlabel("1 / epsilon", fontsize=12)
    plt.ylabel("Average iterations to reach ||z(p^t)||_2 <= epsilon", fontsize=12)
    plt.title(
        "Tatonnement Comparison: Additive vs Multiplicative\n"
        "Average Over 20 Largest Spliddit Instances (rho=2.0, max_steps=100000)",
        fontsize=13,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="best", ncol=2)
    plt.tight_layout()

    plot_path = root / "tatonnement_additive_vs_multiplicative_comparison.png"
    plt.savefig(plot_path, dpi=170)
    plt.close()
    print(f"\n\nSaved plot to: {plot_path}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Average iterations to reach each epsilon")
    print("="*70)
    for eps in epsilons:
        print(f"\nepsilon={eps}  (1/epsilon={1.0/eps:.0f}):")
        print(f"  {'Variant':<15} {'eta=0.001':>12} {'eta=0.002':>12} {'eta=0.005':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        for variant in ["additive", "multiplicative"]:
            row_str = f"  {variant:<15}"
            for eta in eta_values:
                iters = results.get((variant, eta, eps), num_steps + 1)
                row_str += f" {int(iters):>12d}"
            print(row_str)


if __name__ == "__main__":
    main()
