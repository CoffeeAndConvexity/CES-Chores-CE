from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ces_disutility import compute_excess_supply


def _first_hitting_iteration(norms: np.ndarray, epsilon: float) -> int:
    hit_idx = np.where(norms <= epsilon)[0]
    if hit_idx.size == 0:
        return norms.size + 1
    return int(hit_idx[0])

def _simulate_norms(
    method: str,
    prices_init: np.ndarray,
    disutility_weights: np.ndarray,
    earning_requirements: np.ndarray,
    rho: float,
    eta: float,
    num_steps: int,
) -> np.ndarray:
    prices = prices_init.astype(float).copy()
    norms = np.empty(num_steps, dtype=float)

    for t in range(num_steps):
        try:
            z, _ = compute_excess_supply(prices, disutility_weights, earning_requirements, rho)
        except Exception as e:
            print("prices:", prices)
        norms[t] = np.linalg.norm(z)

        if method == "additive":
            update = z - z.mean()
        elif method == "multiplicative":
            u = prices * z
            update = u - u.mean()
        elif method == "quadratic":
            u = (prices ** 2) * z
            update = u - u.mean()
        else:
            raise ValueError(f"Unknown method: {method}")

        prices = np.maximum(prices + eta * update, 1e-12)

    return norms


def _generate_lognormal_instance(
    rng: np.random.Generator,
    n: int,
    m: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Positive disutility matrix and earning requirements.
    D = rng.lognormal(mean=0.0, sigma=1.0, size=(n, m))
    B = np.ones(n, dtype=float)
    p0 = np.full(m, B.sum() / m, dtype=float)
    return D, B, p0


def main() -> None:
    root = Path(__file__).resolve().parent

    n_agents = 100
    n_resources = 200
    num_instances = 5
    seed = 2026

    rho = 5.0
    num_steps = 2000
    epsilons = [0.1, 0.05, 0.01, 0.005, 0.001]

    # Use the currently configured 3 step sizes for each method.
    eta_by_method: dict[str, list[float]] = {
        "additive": [0.05, 0.1, 0.2],
        "multiplicative": [0.2, 0.3, 0.5],
        "quadratic": [0.5, 0.8, 1.0],
    }

    print(
        f"Synthetic benchmark: {num_instances} instances of {n_agents}x{n_resources} "
        f"(lognormal D and B=1), rho={rho}, steps={num_steps}"
    )

    rng = np.random.default_rng(seed)
    instances = [
        _generate_lognormal_instance(rng, n_agents, n_resources)
        for _ in range(num_instances)
    ]

    rows: list[dict[str, float | int | str]] = []

    for method, eta_values in eta_by_method.items():
        print(f"\n=== Method: {method} ===")
        for eta in eta_values:
            print(f"  eta={eta}")
            for instance_idx, (D, B, p0) in enumerate(instances):
                norms = _simulate_norms(
                    method=method,
                    prices_init=p0,
                    disutility_weights=D,
                    earning_requirements=B,
                    rho=rho,
                    eta=eta,
                    num_steps=num_steps,
                )
                print(
                    f"    instance={instance_idx:2d}  ||z_0||={norms[0]:.6f}  "
                    f"||z_T||={norms[-1]:.6f}"
                )

                for eps in epsilons:
                    rows.append(
                        {
                            "method": method,
                            "eta": eta,
                            "instance_idx": instance_idx,
                            "epsilon": eps,
                            "inv_epsilon": 1.0 / eps,
                            "iterations_to_epsilon": _first_hitting_iteration(norms, eps),
                        }
                    )

    df = pd.DataFrame(rows)

    summary = (
        df.groupby(["method", "eta", "epsilon", "inv_epsilon"], as_index=False)["iterations_to_epsilon"]
        .mean()
        .rename(columns={"iterations_to_epsilon": "avg_iterations_to_epsilon"})
    )

    csv_path = root / f"tatonnement_synthetic_100x200_rho{int(rho)}_add_mul_quad_iterations_to_epsilon_avg.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")

    plt.figure(figsize=(13, 8))
    style_by_method = {
        "additive": "-",
        "multiplicative": "--",
        "quadratic": ":",
    }
    color_cycle = {
        ("additive", eta_by_method['additive'][0]): "C0",
        ("additive", eta_by_method['additive'][1]): "C1",
        ("additive", eta_by_method['additive'][2]): "C2",
        ("multiplicative", eta_by_method['multiplicative'][0]): "C3",
        ("multiplicative", eta_by_method['multiplicative'][1]): "C4",
        ("multiplicative", eta_by_method['multiplicative'][2]): "C5",
        ("quadratic", eta_by_method['quadratic'][0]): "C6",
        ("quadratic", eta_by_method['quadratic'][1]): "C7",
        ("quadratic", eta_by_method['quadratic'][2]): "C8",
    }

    for method, eta_values in eta_by_method.items():
        for eta in eta_values:
            sub = summary[(summary["method"] == method) & (summary["eta"] == eta)].sort_values("inv_epsilon")
            plt.plot(
                sub["inv_epsilon"].to_numpy(),
                sub["avg_iterations_to_epsilon"].to_numpy(),
                linestyle=style_by_method[method],
                marker="o",
                linewidth=2,
                markersize=6,
                color=color_cycle[(method, eta)],
                label=f"{method} eta={eta}",
            )

    plt.xscale("log")
    plt.xlabel("1 / epsilon")
    plt.ylabel("Average iterations to reach ||z(p^t)||_2 <= epsilon")
    plt.title(
        "Synthetic 100x200 Lognormal Markets: Additive vs Multiplicative vs Quadratic\n"
        f"({num_instances} instances, rho={rho}, max_steps={num_steps})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()

    plot_path = root / f"tatonnement_synthetic_100x200_rho{int(rho)}_add_mul_quad_iterations_vs_inv_epsilon_avg.png"
    plt.savefig(plot_path, dpi=170)
    plt.close()
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
