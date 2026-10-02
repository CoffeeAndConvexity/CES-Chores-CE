"""Regression checks for experiment behavior and reproducible input handling."""

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import adaptive_stepsize as synthetic
import adaptive_stepsize_bidding as bidding
import adaptive_stepsize_spliddit as spliddit
from experiment_utils import EPSILONS, append_result, run_instance, search_stepsize


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.disutilities = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        self.budgets = np.ones(2)

    def test_original_iteration_counts_and_failure_rule(self):
        # Recorded from the original experiment runner before refactoring.
        cases = {
            "additive": [21, 25, 36, 41, 51],
            "multiplicative": [30, 39, 61, 70, 92],
            "quadratic": [42, 67, None, None, None],
        }
        for method, hits in cases.items():
            with self.subTest(method=method):
                self.assertEqual(
                    run_instance(
                        self.disutilities, self.budgets, method, 0.1, 100, 2.0
                    ),
                    (True, dict(zip(EPSILONS, hits))),
                )
                self.assertEqual(
                    run_instance(
                        self.disutilities, self.budgets, method, 5.0, 100, 2.0
                    ),
                    (False, "negative prices"),
                )

    def test_original_search_results(self):
        expected_hits = dict(zip(EPSILONS, [3, 3, 4, 5, 6]))
        for method, expected_eta in (("additive", 0.63), ("multiplicative", 1.08)):
            with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                eta, data = spliddit.search_largest_stepsize(
                    (self.disutilities, self.budgets),
                    method,
                    max_steps=100,
                )
                self.assertEqual(eta, expected_eta)
                self.assertEqual(data, expected_hits)
                self.assertEqual(
                    bidding.search_largest_stepsize(
                        self.disutilities, method, max_steps=100
                    ),
                    (expected_eta, {"instance": 0, **expected_hits}),
                )

    def test_seeded_synthetic_run_matches_original(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(
                synthetic.search_largest_stepsize(
                    "multiplicative",
                    max_steps=100,
                    instance_seed=0,
                    n_agents=2,
                    n_resources=3,
                ),
                (
                    0.13499999999999998,
                    {"instance": 0, **dict(zip(EPSILONS, [24, 29, 42, 48, 62]))},
                ),
            )

    def test_synthetic_distributions_keep_legacy_random_stream(self):
        # RandomState and np.random.seed share the legacy MT19937 stream. Protect
        # reproducibility against replacing it with default_rng in a later cleanup.
        legacy_draws = {
            "lognormal": lambda rng: rng.lognormal(0.0, 1.0, (3, 4)),
            "uniform": lambda rng: rng.uniform(0.0, 1.0, (3, 4)),
            "integer_uniform": lambda rng: rng.randint(1, 20, (3, 4)),
            "exponential": lambda rng: rng.exponential(1.0, (3, 4)),
            "truncated_normal": lambda rng: np.clip(
                rng.normal(0.5, 0.2, (3, 4)), 0.01, None
            ),
        }
        for distribution, draw in legacy_draws.items():
            with self.subTest(distribution=distribution):
                np.testing.assert_array_equal(
                    synthetic.generate_instance(3, 4, distribution, 17),
                    draw(np.random.RandomState(17)),
                )

    def test_bidding_rejection_and_noise_reseed_match_original(self):
        weights = np.array(
            [[1.0, 1.0, 7.0, 3.0], [7.0, 3.0, 7.0, 3.0], [7.0, 7.0, 7.0, 7.0]]
        )
        distances = bidding.distance_matrix_among_papers(weights)
        # Seed 0 first draws rejected anchor 0, then accepted anchor 3.
        np.testing.assert_array_equal(
            bidding.sample_bidding_instance(weights, distances, 2, 1, 0, 0.0),
            [[3.0], [3.0]],
        )
        np.testing.assert_array_equal(
            bidding.sample_bidding_instance(weights, distances, 2, 1, 0, 1.0),
            [[4.764052345967664], [3.400157208367223]],
        )

    def test_impossible_bidding_sample_fails_instead_of_looping(self):
        weights = np.full((2, 3), 7.0)
        with self.assertRaisesRegex(ValueError, "No anchor"):
            bidding.sample_bidding_instance(weights, np.zeros((3, 3)), 2, 2, 0, 0.0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            bidding.sample_bidding_instance(weights, np.zeros((3, 3)), 3, 2, 0, 0.0)

    def test_bidding_encoding_uses_requested_paper_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bids.csv"
            pd.DataFrame(
                {
                    "Bidder": ["b", "a", "b"],
                    "Submission": [1, 2, 3],
                    "Bid": ["yes", "conflict", "maybe"],
                }
            ).to_csv(path, index=False)
            np.testing.assert_array_equal(
                bidding.load_bidding_data(path, 2),
                [[1.0, 5.0, 3.0], [5.0, 15.0, 5.0]],
            )

    def test_spliddit_metadata_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "instance_2.csv",
                "instance_1.csv",
                "instance_3.csv",
                "_metadata.csv",
            ):
                (root / name).write_text("unused\n", encoding="utf-8")
            index = root / "_index.csv"
            index.write_text(
                "matrix_file\ninstance_2.csv\ninstance_1.csv\n", encoding="utf-8"
            )
            self.assertEqual(
                [path.name for path in spliddit.instance_files(root)],
                ["instance_2.csv", "instance_1.csv", "instance_3.csv"],
            )
            index.unlink()
            self.assertEqual(
                [path.name for path in spliddit.instance_files(root)],
                ["instance_1.csv", "instance_2.csv", "instance_3.csv"],
            )

    def test_invalid_experiment_inputs_fail_clearly(self):
        for max_steps, rho in ((0, 2.0), (1, 1.0), (1, float("nan"))):
            with self.subTest(max_steps=max_steps, rho=rho), self.assertRaises(
                ValueError
            ):
                run_instance(
                    self.disutilities, self.budgets, "additive", 0.1, max_steps, rho
                )
        with self.assertRaisesRegex(RuntimeError, "after 2 rounds"):
            search_stepsize(
                lambda eta: (False, "negative prices"), 1, max_rounds=2, verbose=False
            )

    def test_appending_refuses_to_corrupt_a_different_csv_schema(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(
            io.StringIO()
        ):
            root = Path(directory)
            append_result(root, "results.csv", {"method": "additive", 0.001: 10})
            append_result(root, "results.csv", {"method": "multiplicative", 0.001: 5})
            path = root / "results.csv"
            original = path.read_bytes()
            self.assertEqual(len(pd.read_csv(path)), 2)
            with self.assertRaisesRegex(ValueError, "different columns"):
                append_result(
                    root, "results.csv", {"method": "additive", "matrix_file": "a.csv"}
                )
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
