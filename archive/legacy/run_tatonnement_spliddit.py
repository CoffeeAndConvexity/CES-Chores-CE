from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import run_tatonnement


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


def main() -> None:
    root = Path(__file__).resolve().parent
    matrix_dir = root / "Spliddit_Data" / "distribute_tasks_valuation_matrices"
    index_path = matrix_dir / "_index.csv"

    index_df = pd.read_csv(index_path)
    selected_all = index_df.sort_values("instance_id").reset_index(drop=True)

    # Spread picks across the whole dataset instead of only local neighbors.
    max_instances = 20
    pick_idx = np.linspace(0, len(selected_all) - 1, num=max_instances, dtype=int)
    pick_idx = np.unique(pick_idx)
    selected = selected_all.iloc[pick_idx].copy()

    rho = 2.0
    eta_values = [0.0005, 0.001, 0.002]
    num_steps = 10000

    print(
        f"Running {len(selected)} instances: "
        + ", ".join(str(int(x)) for x in selected["instance_id"].tolist())
    )

    for eta in eta_values:
        plt.figure(figsize=(10, 6))

        for _, row in selected.iterrows():
            instance_id = int(row["instance_id"])
            matrix_file = str(row["matrix_file"])

            D = _load_disutility_matrix(matrix_dir / matrix_file)
            n, m = D.shape

            # Equal earning requirements; initialize prices so sum(p0) = sum(B).
            B = np.ones(n, dtype=float)
            p0 = np.full(m, B.sum() / m, dtype=float)

            prices, excess_supplies = run_tatonnement(
                prices_init=p0,
                disutility_weights=D,
                earning_requirements=B,
                rho=rho,
                eta=eta,
                num_steps=num_steps,
            )

            # Keep this variable to make it easy to inspect if needed.
            _ = prices

            norms = np.linalg.norm(excess_supplies, axis=1)
            plt.plot(norms, label=f"instance {instance_id} (n={n}, m={m})")

            print(
                f"eta={eta:0.4f}  instance={instance_id:5d}  n={n:2d}  m={m:2d}  "
                f"||z_0||={norms[0]:.6f}  ||z_T||={norms[-1]:.6f}"
            )

        plt.xlabel("Tatonnement step t")
        plt.ylabel("||z(p^t)||_2")
        plt.title(
            f"Tatonnement on Spliddit Chores Instances (rho={rho}, eta={eta}, steps={num_steps})"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        eta_tag = str(eta).replace(".", "p")
        out_path = root / f"tatonnement_excess_norms_eta{eta_tag}_steps{num_steps}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
