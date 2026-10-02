from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent
    csv_path = root / "tatonnement_iterations_to_epsilon_eta0p001_steps10000.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            "Missing epsilon-hitting CSV. Run run_tatonnement_epsilon_hitting.py first."
        )

    df = pd.read_csv(csv_path)
    required_cols = {"instance_id", "epsilon", "iterations_to_epsilon"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV is missing required columns: {sorted(required_cols)}")

    df = df.copy()
    df["inv_epsilon"] = 1.0 / df["epsilon"]

    plt.figure(figsize=(12, 7))

    for instance_id, sub in df.groupby("instance_id"):
        s = sub.sort_values("inv_epsilon")
        plt.plot(
            s["inv_epsilon"].to_numpy(),
            s["iterations_to_epsilon"].to_numpy(),
            marker="o",
            linewidth=1.2,
            markersize=4,
            alpha=0.85,
            label=f"instance {int(instance_id)}",
        )

    plt.xscale("log")
    plt.xlabel("1 / epsilon")
    plt.ylabel("Iterations to first reach ||z(p^t)||_2 <= epsilon")
    plt.title("Iterations Required vs 1/epsilon (one curve per instance)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    out_path = root / "tatonnement_iterations_vs_inv_epsilon_by_instance.png"
    plt.savefig(out_path, dpi=170)
    plt.close()
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()