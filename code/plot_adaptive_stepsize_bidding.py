"""Plot adaptive stepsize convergence on the bidding dataset."""

import argparse

import pandas as pd

from plotting import (
    METHODS,
    IterationSeries,
    add_path_arguments,
    plot_iteration_comparison,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_arguments(parser)
    parser.add_argument(
        "--missing-iterations",
        type=float,
        default=10000,
        help="Iteration budget used for missing convergence values (default: 10000).",
    )
    args = parser.parse_args(argv)
    series = [
        IterationSeries(
            method,
            pd.read_csv(
                args.data_dir / f"batch_adaptive_stepsize_{method}_bidding.csv"
            ),
        )
        for method in METHODS
    ]
    plot_iteration_comparison(
        series,
        args.output_dir / "bidding.pdf",
        missing_iterations=args.missing_iterations,
        marker_size=10,
    )


if __name__ == "__main__":
    main()
