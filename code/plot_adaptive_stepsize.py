"""Plot adaptive stepsize convergence for synthetic disutility matrices."""

import argparse

import pandas as pd

from plotting import (
    METHODS,
    IterationSeries,
    add_path_arguments,
    plot_iteration_comparison,
)


N_RESOURCES_LINESTYLE = {500: "solid", 1000: "dashed", 2000: "dotted"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    parser.add_argument(
        "--distribution",
        default="lognormal",
        choices=("lognormal", "integer_uniform", "truncated_normal"),
        help="Synthetic value distribution (default: lognormal).",
    )
    args = parser.parse_args(argv)
    series = []
    for n_resources, linestyle in N_RESOURCES_LINESTYLE.items():
        for method in METHODS:
            csv_path = args.data_dir / (
                f"batch_adaptive_stepsize_{method}_100_{n_resources}_{args.distribution}.csv"
            )
            series.append(IterationSeries(method, pd.read_csv(csv_path), linestyle))
    plot_iteration_comparison(series, args.output_dir / f"{args.distribution}.pdf")


if __name__ == "__main__":
    main()
