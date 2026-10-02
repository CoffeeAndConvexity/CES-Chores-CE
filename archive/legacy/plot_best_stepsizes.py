from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent
    csv_path = root / "tatonnement_largest_multistepsize_iterations_to_epsilon_avg.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter to only the best three step sizes
    best_etas = [0.001, 0.002, 0.005]
    df_filtered = df[df["eta"].isin(best_etas)].copy()

    if df_filtered.empty:
        raise ValueError(f"No data found for eta values: {best_etas}")

    # Create the plot
    plt.figure(figsize=(11, 7))

    colors = {0.001: "C0", 0.002: "C1", 0.005: "C2"}
    markers = {0.001: "o", 0.002: "s", 0.005: "^"}

    for eta in best_etas:
        sub = df_filtered[df_filtered["eta"] == eta].sort_values("inv_epsilon")
        plt.plot(
            sub["inv_epsilon"].to_numpy(),
            sub["avg_iterations_to_epsilon"].to_numpy(),
            marker=markers[eta],
            linewidth=2.5,
            markersize=8,
            color=colors[eta],
            label=f"eta={eta}",
        )

    plt.xscale("log")
    plt.xlabel("1 / epsilon", fontsize=12)
    plt.ylabel("Average iterations to reach ||z(p^t)||_2 <= epsilon", fontsize=12)
    plt.title(
        "Tatonnement Convergence: Best Step Sizes\n"
        "Average Over 20 Largest Spliddit Instances (rho=2.0, max_steps=100000)",
        fontsize=13,
    )
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc="best")
    plt.tight_layout()

    out_path = root / "tatonnement_largest_best_stepsizes_iterations_vs_inv_epsilon_avg.png"
    plt.savefig(out_path, dpi=170)
    plt.close()
    print(f"Saved plot to: {out_path}")

    # Print summary table
    print("\nAverage iterations to reach each epsilon (best 3 step sizes):")
    print("=" * 65)
    for eps in [0.1, 0.05, 0.01, 0.005, 0.001]:
        print(f"epsilon={eps}:")
        for eta in best_etas:
            row = df_filtered[(df_filtered["eta"] == eta) & (df_filtered["epsilon"] == eps)]
            if not row.empty:
                iters = row["avg_iterations_to_epsilon"].values[0]
                print(f"  eta={eta}: {int(iters):6d} iterations")
        print()


if __name__ == "__main__":
    main()
