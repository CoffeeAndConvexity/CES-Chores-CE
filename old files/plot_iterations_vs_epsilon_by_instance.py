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
    required_cols = {
        "instance_id",
        "epsilon",
        "iterations_to_epsilon",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV is missing required columns: {sorted(required_cols)}")

    # Keep epsilon order from coarse to fine for readability.
    eps_order = [0.1, 0.05, 0.01, 0.005, 0.001]

    plt.figure(figsize=(12, 7))

    for instance_id, sub in df.groupby("instance_id"):
        s = sub.copy()
        s["epsilon"] = pd.Categorical(s["epsilon"], categories=eps_order, ordered=True)
        s = s.sort_values("epsilon")

        plt.plot(
            s["epsilon"].astype(float).to_numpy(),
            s["iterations_to_epsilon"].to_numpy(),
            marker="o",
            linewidth=1.2,
            markersize=4,
            alpha=0.85,
            label=f"instance {int(instance_id)}",
        )

    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xticks(eps_order, [str(e) for e in eps_order])
    plt.xlabel("epsilon")
    plt.ylabel("Iterations to first reach ||z(p^t)||_2 <= epsilon")
    plt.title("Iterations Required vs Epsilon (one curve per instance)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    out_path = root / "tatonnement_iterations_vs_epsilon_by_instance.png"
    plt.savefig(out_path, dpi=170)
    plt.close()
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()