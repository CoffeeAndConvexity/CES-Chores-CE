"""Shared styles and statistics for the adaptive stepsize figures."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "data" / "results"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"
METHODS = ("additive", "multiplicative")
EPSILONS = (0.1, 0.05, 0.01, 0.005, 0.001)
RHO_VALUES = (1.2, 2.0, 5.0)
COLOR_MAP = {"additive": "#b3480e", "multiplicative": "#0e5195"}
SHAPE_MAP = {"additive": "s", "multiplicative": "^"}
FONTSIZE = 24


@dataclass
class IterationSeries:
    method: str
    data: pd.DataFrame
    linestyle: str = "solid"


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing experiment result CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Directory for generated figures.",
    )


def iteration_statistics(
    data: pd.DataFrame, missing_iterations: float | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inverse tolerances, means and sample standard deviations.

    Synthetic results omit missing entries. Bidding and Spliddit use their
    historical iteration budgets when a tolerance was not reached.
    """
    columns = [str(epsilon) for epsilon in EPSILONS]
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing epsilon columns: {', '.join(missing_columns)}")
    iterations = data[columns]
    if missing_iterations is not None:
        iterations = iterations.fillna(missing_iterations)
    return (
        1 / np.asarray(EPSILONS),
        iterations.mean().to_numpy(),
        iterations.std().to_numpy(),
    )


def plot_iteration_comparison(
    series: Sequence[IterationSeries],
    output_path: Path,
    *,
    missing_iterations: float | None = None,
    errorbar_scale: float = 1.0,
    marker_size: int = 12,
    ytick_rotation: int = 0,
) -> None:
    """Plot the three historical rho cases with the original figure styling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    try:
        for index, (axis, rho) in enumerate(zip(axes, RHO_VALUES)):
            labeled_methods = set()
            for curve in series:
                data_rho = curve.data.loc[curve.data["rho"] == rho]
                if data_rho.empty:
                    print(f"No data for rho={rho}, method={curve.method}")
                    continue
                inverse_eps, means, deviations = iteration_statistics(
                    data_rho, missing_iterations
                )
                label = curve.method if curve.method not in labeled_methods else None
                axis.errorbar(
                    inverse_eps,
                    means,
                    yerr=errorbar_scale * deviations,
                    label=label,
                    marker=SHAPE_MAP[curve.method],
                    capsize=3,
                    markersize=marker_size,
                    linewidth=3,
                    color=COLOR_MAP[curve.method],
                    linestyle=curve.linestyle,
                )
                axis.scatter(
                    inverse_eps,
                    means,
                    marker=SHAPE_MAP[curve.method],
                    color=COLOR_MAP[curve.method],
                )
                labeled_methods.add(curve.method)

            axis.set_xscale("log")
            axis.set_xlabel(r"$1 / \epsilon$", fontsize=FONTSIZE)
            if index == 0:
                axis.set_ylabel("Average iterations", fontsize=FONTSIZE)
            axis.tick_params(axis="x", labelsize=FONTSIZE)
            axis.tick_params(axis="y", labelsize=FONTSIZE, labelrotation=ytick_rotation)
            axis.locator_params(axis="y", nbins=5)
            axis.set_title(rf"$\rho={rho}$", fontsize=FONTSIZE)
            axis.grid(True, alpha=0.3)
            if labeled_methods:
                axis.legend(fontsize=FONTSIZE)

        fig.tight_layout()
        fig.savefig(
            output_path,
            facecolor="white",
            transparent=False,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.03,
            dpi=170,
        )
    finally:
        plt.close(fig)
    print(f"Wrote {output_path}")
