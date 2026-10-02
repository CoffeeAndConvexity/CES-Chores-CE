from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import compute_excess_supply


def simulate_tatonnement(
    prices_init: np.ndarray,
    disutility_weights: np.ndarray,
    earning_requirements: np.ndarray,
    rho: float,
    num_steps: int,
    step_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate tatonnement with a generic step-size schedule.

    Update rule:
        p^{t+1} = max(p^t + eta_t * (z(p^t) - mean(z(p^t))), 1e-12)
    """
    p = prices_init.astype(float).copy()
    m = p.size
    prices = np.empty((num_steps + 1, m), dtype=float)
    excess_norms = np.empty(num_steps, dtype=float)

    prices[0] = p
    for t in range(num_steps):
        z = compute_excess_supply(p, disutility_weights, earning_requirements, rho)
        excess_norms[t] = np.linalg.norm(z)
        relative_z = z - z.mean()
        eta_t = float(step_fn(t))
        if eta_t <= 0:
            raise ValueError(f"step size must be positive, got {eta_t} at t={t}")
        p = np.maximum(p + eta_t * relative_z, 1e-12)
        prices[t + 1] = p

    return prices, excess_norms


def slugify_setting(name: str) -> str:
    out = name.lower().replace(" ", "_").replace("=", "")
    out = out.replace("/", "_over_").replace("^", "pow")
    out = out.replace("(", "").replace(")", "")
    out = out.replace(".", "p")
    return out


def main() -> None:
    root = Path(__file__).resolve().parent
    matrix_path = (
        root
        / "Spliddit_Data"
        / "distribute_tasks_valuation_matrices"
        / "instance_10513_valuation_matrix.csv"
    )

    df = pd.read_csv(matrix_path)
    item_cols = [c for c in df.columns if c != "agent_id"]
    D = df[item_cols].to_numpy(dtype=float)

    n, m = D.shape
    B = np.ones(n, dtype=float)
    p0 = np.full(m, B.sum() / m, dtype=float)

    rho = 2.0
    num_steps = 500000

    # Requested experiments: larger fixed steps and shrinking-step schedules.
    settings = [
        ("fixed eta=0.01", lambda t: 0.01),
        ("fixed eta=0.02", lambda t: 0.02),
        ("fixed eta=0.05", lambda t: 0.05),
        ("shrinking eta_t=0.05/sqrt(t+1)", lambda t: 0.05 / np.sqrt(t + 1.0)),
        ("shrinking eta_t=0.10/sqrt(t+1)", lambda t: 0.10 / np.sqrt(t + 1.0)),
        ("shrinking eta_t=0.05/(t+1)", lambda t: 0.05 / (t + 1.0)),
    ]

    for name, step_fn in settings:
        prices, norms = simulate_tatonnement(
            prices_init=p0,
            disutility_weights=D,
            earning_requirements=B,
            rho=rho,
            num_steps=num_steps,
            step_fn=step_fn,
        )

        plt.figure(figsize=(10, 6))
        plt.plot(norms, linewidth=1.6)
        plt.xlabel("Tatonnement step t")
        plt.ylabel("||z(p^t)||_2")
        plt.title(f"Instance 10513: {name} (rho={rho}, steps={num_steps})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_name = f"tatonnement_10513_{slugify_setting(name)}_steps{num_steps}.png"
        out_path = root / out_name
        plt.savefig(out_path, dpi=160)
        plt.close()

        p_final = prices[-1]
        print(
            f"{name:34s}  ||z_0||={norms[0]:.6f}  ||z_T||={norms[-1]:.6f}  "
            f"sum(p_T)={p_final.sum():.12f}  file={out_name}"
        )


if __name__ == "__main__":
    main()
