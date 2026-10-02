from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import run_quadratic_price_tatonnement


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
    eta_values = [0.2, 0.3, 0.4]
    num_steps = 5000
    epsilons = [0.1, 0.05, 0.01, 0.005, 0.001]

    print(
        f"Running quadratic-price tatonnement on {len(chosen_ids)} largest instances: {chosen_ids}"
    )
    print(f"Parameters: rho={rho}, num_steps={num_steps}")
    print(f"Testing {len(eta_values)} step sizes: {eta_values}")

    avg_iterations: dict[tuple[float, float], float] = {}

    for eta in eta_values:
        print(f"\n--- eta={eta} ---")
        rows_this_eta: list[dict[str, float | int]] = []

        for instance_id in chosen_ids:
            row = index_df[index_df["instance_id"] == instance_id].iloc[0]
            matrix_file = str(row["matrix_file"])

            D = _load_disutility_matrix(matrix_dir / matrix_file)
            n, m = D.shape
            B = np.ones(n, dtype=float)
            p0 = np.full(m, B.sum() / m, dtype=float)

            _, excess_supplies = run_quadratic_price_tatonnement(
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
                rows_this_eta.append(
                    {
                        "epsilon": eps,
                        "iterations_to_epsilon": t_hit,
                    }
                )

        df_this_eta = pd.DataFrame(rows_this_eta)
        for eps in epsilons:
            avg_iters = df_this_eta[df_this_eta["epsilon"] == eps]["iterations_to_epsilon"].mean()
            avg_iterations[(eta, eps)] = avg_iters

    plt.figure(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(eta_values)))

    for idx, eta in enumerate(eta_values):
        inv_eps = 1.0 / np.array(epsilons)
        avg_iters = np.array([avg_iterations.get((eta, eps), num_steps + 1) for eps in epsilons])

        plt.plot(
            inv_eps,
            avg_iters,
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors[idx],
            label=f"eta={eta}",
        )

    plt.xscale("log")
    plt.xlabel("1 / epsilon")
    plt.ylabel("Average iterations to reach ||z(p^t)||_2 <= epsilon")
    plt.title(
        "Quadratic-Price Tatonnement: Average Over 20 Largest Instances\n"
        "10 Step Sizes (rho=2.0, max_steps=100000)"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    plot_path = root / "tatonnement_quadratic_price_largest_multistepsize_iterations_vs_inv_epsilon_avg.png"
    plt.savefig(plot_path, dpi=170)
    plt.close()
    print(f"\nSaved plot to: {plot_path}")

    csv_rows = []
    for eta in eta_values:
        for eps in epsilons:
            avg_iters = avg_iterations.get((eta, eps), num_steps + 1)
            csv_rows.append(
                {
                    "eta": eta,
                    "epsilon": eps,
                    "inv_epsilon": 1.0 / eps,
                    "avg_iterations_to_epsilon": avg_iters,
                }
            )

    csv_df = pd.DataFrame(csv_rows)
    csv_path = root / "tatonnement_quadratic_price_largest_multistepsize_iterations_to_epsilon_avg.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()