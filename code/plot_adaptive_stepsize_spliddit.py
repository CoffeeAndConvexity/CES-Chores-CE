"""Plot adaptive stepsize convergence on the Spliddit dataset."""

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
        default=60000,
        help="Iteration budget used for missing convergence values (default: 60000).",
    )
    args = parser.parse_args(argv)
    series = [
        IterationSeries(
            method,
            pd.read_csv(
                args.data_dir / f"batch_adaptive_stepsize_{method}_spliddit.csv"
            ),
        )
        for method in METHODS
    ]
    # Preserve the published figure's error bars: one tenth of the sample std.
    plot_iteration_comparison(
        series,
        args.output_dir / "spliddit.pdf",
        missing_iterations=args.missing_iterations,
        errorbar_scale=0.1,
        ytick_rotation=90,
    )


if __name__ == "__main__":
    main()
