from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd


COLOR_MAP = {
    "additive": "#b3480e",
    "multiplicative": "#0e5195",
}

SHAPE_MAP = {
    "additive": "s",
    "multiplicative": "^",
}

FONTSIZE = 24


def _power_of_ten_ticks(max_value: float) -> tuple[list[float], list[str]]:
    max_exponent = int(np.ceil(np.log10(max_value)))
    exponents = range(0, max_exponent + 1)
    ticks = [10.0 ** exponent for exponent in exponents]
    labels = [rf"$10^{exponent}$" for exponent in exponents]
    return ticks, labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best stepsize and convergence iterations against Spliddit instance value ratios."
    )
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--missing-iterations", type=float, default=60000)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=Path("Spliddit_Data/distribute_tasks_valuation_matrices"),
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("spliddit_ratio_relations"))
    parser.add_argument("--rho", type=float, default=5.0)
    parser.add_argument("--show-slopes", action="store_true")
    parser.add_argument(
        "--fit-ratio-max",
        type=float,
        default=50.0,
        help="Use only instances with value ratio at most this number for the additive exponent estimate.",
    )
    parser.add_argument(
        "--plot-converged-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot only runs that reached the target epsilon instead of budget-filled nonconverged runs.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the merged ratio CSV from raw Spliddit matrices instead of reusing it if present.",
    )
    parser.add_argument(
        "--show-additive-estimate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the additive log-log exponent estimate on the clean default plot.",
    )
    parser.add_argument(
        "--use-index-order",
        action="store_true",
        help=(
            "Use _index.csv order for instance_idx. By default this script uses os.listdir order, "
            "matching the current adaptive_stepsize_spliddit.py outputs."
        ),
    )
    return parser.parse_args()


def _instance_files(matrix_dir: Path, use_index_order: bool) -> list[str]:
    if use_index_order:
        index_path = matrix_dir / "_index.csv"
        index_df = pd.read_csv(index_path)
        return index_df["matrix_file"].tolist()

    return [
        filename
        for filename in os.listdir(matrix_dir)
        if filename.endswith(".csv") and not filename.startswith("_")
    ]


def _value_ratio(values: np.ndarray) -> float:
    positive_values = values[np.isfinite(values) & (values > 0)]
    if positive_values.size == 0:
        return np.nan
    return float(positive_values.max() / positive_values.min())


def build_ratio_table(matrix_dir: Path, use_index_order: bool) -> pd.DataFrame:
    rows = []
    for instance_idx, filename in enumerate(_instance_files(matrix_dir, use_index_order)):
        matrix_path = matrix_dir / filename
        D = pd.read_csv(matrix_path, index_col=0).values.astype(float)
        positive_values = D[np.isfinite(D) & (D > 0)]
        rows.append(
            {
                "instance_idx": instance_idx,
                "matrix_file": filename,
                "n_agents": D.shape[0],
                "n_resources": D.shape[1],
                "min_positive_value": positive_values.min() if positive_values.size else np.nan,
                "max_value": positive_values.max() if positive_values.size else np.nan,
                "max_value_ratio": _value_ratio(D),
            }
        )
    return pd.DataFrame(rows)


def load_experiment_data(data_dir: Path, epsilon: float, missing_iterations: float) -> pd.DataFrame:
    eps_col = str(epsilon)
    frames = []
    for method in ["additive", "multiplicative"]:
        csv_path = data_dir / f"batch_adaptive_stepsize_{method}_spliddit.csv"
        df = pd.read_csv(csv_path)
        if eps_col not in df.columns:
            raise ValueError(f"{csv_path} does not contain epsilon column {eps_col!r}.")
        df = df.copy()
        df["iterations"] = df[eps_col].fillna(missing_iterations)
        df["converged"] = df[eps_col].notna()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _save_rgb_png(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    fig.savefig(
        tmp_path,
        dpi=dpi,
        facecolor="white",
        transparent=False,
        format="png",
        bbox_inches="tight",
        pad_inches=0.03,
    )

    # Some IDE image viewers struggle with RGBA PNGs. Convert to RGB after saving.
    try:
        from PIL import Image

        with Image.open(tmp_path) as image:
            image.convert("RGB").save(tmp_path, format="PNG")
    except ImportError:
        pass
    os.replace(tmp_path, output_path)


def plot_relations(
    df: pd.DataFrame,
    output_path: Path,
    epsilon: float,
    rho: float,
    show_slopes: bool,
    show_additive_estimate: bool,
    fit_ratio_max: float,
    plot_converged_only: bool,
) -> None:
    df = df.copy()
    df["iterations_for_plot"] = df["iterations"].clip(lower=1)
    df["eta_for_plot"] = df["eta"].clip(lower=np.nextafter(0, 1))
    df["ratio_for_plot"] = df["max_value_ratio"].clip(lower=np.nextafter(0, 1))
    ratio_ticks, ratio_tick_labels = _power_of_ten_ticks(float(df["ratio_for_plot"].max()))

    eta_rhos = [1.2, 2.0, rho]
    fig = plt.figure(figsize=(18, 6))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.42, 1], wspace=0.0)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[0, 4]),
    ]
    axes[1].sharey(axes[0])
    axes[2].sharey(axes[0])

    for idx, eta_rho in enumerate(eta_rhos):
        df_rho = df.loc[np.isclose(df["rho"], eta_rho)].copy()
        if plot_converged_only:
            df_rho = df_rho.loc[df_rho["converged"]].copy()
        if df_rho.empty:
            raise ValueError(
                f"No plotted data found for rho={eta_rho}. Available rho values: {sorted(df['rho'].unique())}"
            )

        for method in ["additive", "multiplicative"]:
            df_method = df_rho.loc[df_rho["method"] == method]
            axes[idx].scatter(
                df_method["ratio_for_plot"],
                df_method["eta_for_plot"],
                label=method,
                marker=SHAPE_MAP[method],
                color=COLOR_MAP[method],
                alpha=0.65,
                s=30,
            )
        axes[idx].set_title(rf"$\rho={eta_rho}$", fontsize=FONTSIZE)

    df_rho = df.loc[np.isclose(df["rho"], rho)].copy()
    if plot_converged_only:
        df_rho = df_rho.loc[df_rho["converged"]].copy()
    if df_rho.empty:
        raise ValueError(
            f"No plotted data found for rho={rho}. Available rho values: {sorted(df['rho'].unique())}"
        )

    for method in ["additive", "multiplicative"]:
        df_method = df_rho.loc[df_rho["method"] == method]
        axes[3].scatter(
            df_method["ratio_for_plot"],
            df_method["iterations_for_plot"],
            label=method,
            marker=SHAPE_MAP[method],
            color=COLOR_MAP[method],
            alpha=0.65,
            s=30,
        )

    axes[0].set_ylabel("Best stepsize", fontsize=FONTSIZE)
    axes[0].yaxis.labelpad = 8
    axes[3].set_ylabel(rf"Iterations to $\epsilon={epsilon}$", fontsize=FONTSIZE)
    axes[3].yaxis.labelpad = 10
    axes[3].set_title(rf"$\rho={rho}$", fontsize=FONTSIZE)

    for ax in axes[:3]:
        ax.set_xlabel(rf"$\max_{{i,j,j'}} d_{{ij}} / d_{{ij'}}$", fontsize=FONTSIZE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ratio_ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(ratio_tick_labels))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2.0, 10.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=FONTSIZE - 2)
        ax.tick_params(axis="y", labelsize=FONTSIZE - 2)

    for ax in axes[1:3]:
        ax.tick_params(labelleft=False)

    axes[3].set_xlabel(rf"$\max_{{i,j,j'}} d_{{ij}} / d_{{ij'}}$", fontsize=FONTSIZE)
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].xaxis.set_major_locator(FixedLocator(ratio_ticks))
    axes[3].xaxis.set_major_formatter(FixedFormatter(ratio_tick_labels))
    axes[3].xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2.0, 10.0)))
    axes[3].xaxis.set_minor_formatter(NullFormatter())
    axes[3].grid(True, alpha=0.3)
    axes[3].tick_params(axis="x", labelsize=FONTSIZE - 6)
    axes[3].tick_params(axis="y", labelsize=FONTSIZE - 2)

    axes[0].legend(fontsize=FONTSIZE - 2)
    axes[3].legend(fontsize=FONTSIZE - 2)

    for ax in axes:
        ax.set_xlabel(rf"$\max_{{i,j,j'}} d_{{ij}} / d_{{ij'}}$", fontsize=FONTSIZE)

    # title_suffix = ", converged only" if plot_converged_only else ""
    # fig.suptitle(rf"Spliddit instances, $\rho={rho}$" + title_suffix, fontsize=FONTSIZE + 2)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.91, wspace=0.0)
    _save_rgb_png(fig, output_path, dpi=140)

    preview_path = output_path.with_name(f"{output_path.stem}_preview{output_path.suffix}")
    _save_rgb_png(fig, preview_path, dpi=85)

    pdf_path = output_path.with_suffix(".pdf")
    tmp_pdf_path = pdf_path.with_name(f".{pdf_path.name}.tmp")
    fig.savefig(
        tmp_pdf_path,
        facecolor="white",
        transparent=False,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    os.replace(tmp_pdf_path, pdf_path)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    csv_path = args.output_prefix.with_suffix(".csv")
    png_path = args.output_prefix.with_suffix(".png")

    if csv_path.exists() and not args.rebuild:
        merged = pd.read_csv(csv_path)
        eps_col = str(args.epsilon)
        if eps_col in merged.columns:
            merged["iterations"] = merged[eps_col].fillna(args.missing_iterations)
            merged["converged"] = merged[eps_col].notna()
    else:
        ratio_df = build_ratio_table(args.matrix_dir, args.use_index_order)
        experiment_df = load_experiment_data(args.data_dir, args.epsilon, args.missing_iterations)
        merged = experiment_df.merge(ratio_df, on="instance_idx", how="left", validate="many_to_one")
        tmp_csv_path = csv_path.with_name(f".{csv_path.name}.tmp")
        merged.to_csv(tmp_csv_path, index=False)
        os.replace(tmp_csv_path, csv_path)
    plot_relations(
        merged,
        png_path,
        args.epsilon,
        args.rho,
        args.show_slopes,
        args.show_additive_estimate,
        args.fit_ratio_max,
        args.plot_converged_only,
    )

    summary = (
        merged.groupby(["rho", "method"])
        .agg(
            num_instances=("instance_idx", "count"),
            convergence_rate=("converged", "mean"),
            median_eta=("eta", "median"),
            median_iterations=("iterations", "median"),
            median_ratio=("max_value_ratio", "median"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
