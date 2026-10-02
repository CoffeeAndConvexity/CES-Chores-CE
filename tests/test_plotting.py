"""Regression checks for plot statistics and historical Spliddit joins."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from plotting import EPSILONS, iteration_statistics  # noqa: E402
from plot_spliddit_ratio_relations import merge_ratio_data  # noqa: E402


class PlotStatisticsTests(unittest.TestCase):
    def test_missing_values_and_sample_standard_deviation(self):
        frame = pd.DataFrame({str(epsilon): [1.0, 3.0, np.nan] for epsilon in EPSILONS})
        inverse, means, deviations = iteration_statistics(frame)
        np.testing.assert_array_equal(inverse, [10, 20, 100, 200, 1000])
        np.testing.assert_allclose(means, 2.0)
        np.testing.assert_allclose(deviations, np.sqrt(2))

        _, means, deviations = iteration_statistics(frame, missing_iterations=5)
        np.testing.assert_allclose(means, 3.0)
        np.testing.assert_allclose(deviations, 2.0)


class SplidditRatioJoinTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.matrix_dir = Path(self.directory.name)
        pd.DataFrame([[1.0, 2.0]]).to_csv(self.matrix_dir / "first.csv")
        pd.DataFrame([[1.0, 9.0]]).to_csv(self.matrix_dir / "second.csv")

    def test_new_results_use_filenames_even_when_numeric_ids_change(self):
        experiments = pd.DataFrame(
            {"instance_idx": [0, 1], "matrix_file": ["second.csv", "first.csv"]}
        )
        historical = pd.DataFrame(
            {"instance_idx": [0, 1], "matrix_file": ["first.csv", "second.csv"]}
        )
        merged = merge_ratio_data(experiments, self.matrix_dir, historical)
        self.assertEqual(merged["max_value_ratio"].tolist(), [9.0, 2.0])

    def test_historical_results_use_saved_mapping(self):
        experiments = pd.DataFrame({"instance_idx": [0, 1, 0]})
        historical = pd.DataFrame(
            {
                "instance_idx": [0, 1, 0],
                "matrix_file": ["second.csv", "first.csv", "second.csv"],
            }
        )
        merged = merge_ratio_data(experiments, self.matrix_dir, historical)
        self.assertEqual(merged["max_value_ratio"].tolist(), [9.0, 2.0, 9.0])

    def test_historical_results_require_an_unambiguous_mapping(self):
        experiments = pd.DataFrame({"instance_idx": [0, 1]})
        with self.assertRaisesRegex(ValueError, "original instance_idx"):
            merge_ratio_data(experiments, self.matrix_dir)

        conflicting = pd.DataFrame(
            {"instance_idx": [0, 0], "matrix_file": ["first.csv", "second.csv"]}
        )
        with self.assertRaisesRegex(ValueError, "conflicting filenames"):
            merge_ratio_data(experiments, self.matrix_dir, conflicting)

        incomplete = pd.DataFrame({"instance_idx": [0], "matrix_file": ["first.csv"]})
        with self.assertRaisesRegex(ValueError, "does not map every"):
            merge_ratio_data(experiments, self.matrix_dir, incomplete)


if __name__ == "__main__":
    unittest.main()
