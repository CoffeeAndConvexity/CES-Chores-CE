"""Economic invariants and regression checks for the public numerical routines."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from ces_disutility import (  # noqa: E402
    compute_ces_demand,
    compute_ces_disutility,
    compute_excess_supply,
    run_multiplicative_tatonnement,
    run_quadratic_price_tatonnement,
    run_tatonnement,
)


class CESDisutilityTests(unittest.TestCase):
    def test_disutility_is_homogeneous_and_linear_at_rho_one(self):
        allocation = np.array([0.4, 1.2, 0.8])
        weights = np.array([1.0, 3.0, 2.0])
        self.assertAlmostEqual(
            compute_ces_disutility(allocation, weights, 1.0),
            float(weights @ allocation),
        )
        for rho in (1.2, 2.0, 5.0):
            with self.subTest(rho=rho):
                self.assertAlmostEqual(
                    compute_ces_disutility(3.0 * allocation, weights, rho),
                    3.0 * compute_ces_disutility(allocation, weights, rho),
                )

    def test_demand_meets_earning_requirement(self):
        prices = np.array([0.5, 2.0, 1.5])
        weights = np.array([1.0, 3.0, 2.0])
        for rho in (1.2, 2.0, 5.0):
            with self.subTest(rho=rho):
                demand = compute_ces_demand(prices, weights, 2.5, rho)
                self.assertTrue(np.all(demand >= 0.0))
                self.assertAlmostEqual(float(prices @ demand), 2.5)

    def test_demand_is_unchanged_by_currency_and_preference_scaling(self):
        prices = np.array([0.5, 2.0, 1.5])
        weights = np.array([1.0, 3.0, 2.0])
        demand = compute_ces_demand(prices, weights, 2.5, 2.0)
        np.testing.assert_allclose(
            compute_ces_demand(7.0 * prices, weights, 7.0 * 2.5, 2.0),
            demand,
        )
        np.testing.assert_allclose(
            compute_ces_demand(prices, 4.0 * weights, 2.5, 2.0),
            demand,
        )

    def test_demand_minimizes_disutility_over_feasible_perturbations(self):
        prices = np.array([0.5, 2.0, 1.5])
        weights = np.array([1.0, 3.0, 2.0])
        for rho in (1.2, 2.0, 5.0):
            with self.subTest(rho=rho):
                demand = compute_ces_demand(prices, weights, 2.5, rho)
                optimum = compute_ces_disutility(demand, weights, rho)
                # Transfer earnings between each pair of chores while preserving
                # feasibility; neither direction can improve the optimum.
                for first in range(prices.size):
                    for second in range(first + 1, prices.size):
                        amount = 0.1 * min(
                            demand[first] * prices[first],
                            demand[second] * prices[second],
                        )
                        for sign in (-1, 1):
                            candidate = demand.copy()
                            candidate[first] += sign * amount / prices[first]
                            candidate[second] -= sign * amount / prices[second]
                            self.assertAlmostEqual(float(prices @ candidate), 2.5)
                            self.assertGreaterEqual(
                                compute_ces_disutility(candidate, weights, rho),
                                optimum - 1e-12,
                            )

    def test_zero_cost_chore_and_zero_earning_requirement(self):
        prices = np.array([0.0, 2.0, 1.0])
        weights = np.array([1.0, 0.0, 3.0])
        demand = compute_ces_demand(prices, weights, 4.0, 2.0)
        self.assertAlmostEqual(float(prices @ demand), 4.0)
        self.assertEqual(compute_ces_disutility(demand, weights, 2.0), 0.0)
        np.testing.assert_array_equal(
            compute_ces_demand(prices, weights, 0.0, 2.0), np.zeros(3)
        )

    def test_excess_supply_agrees_with_individual_demands_and_walras_law(self):
        prices = np.array([0.5, 1.25, 1.75])
        weights = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 1.0]])
        earnings = np.array([1.0, 2.5])
        for rho in (1.2, 2.0, 5.0):
            with self.subTest(rho=rho):
                excess = compute_excess_supply(prices, weights, earnings, rho)
                demands = np.array(
                    [
                        compute_ces_demand(prices, row, earning, rho)
                        for row, earning in zip(weights, earnings)
                    ]
                )
                np.testing.assert_allclose(excess, 1.0 - demands.sum(axis=0))
                self.assertAlmostEqual(float(prices @ excess), 0.0)

    def test_symmetric_market_clears_at_equal_prices(self):
        excess = compute_excess_supply(
            np.full(4, 0.75), np.ones((3, 4)), np.ones(3), 2.0
        )
        np.testing.assert_allclose(excess, 0.0, atol=1e-14)

    def test_short_trajectories_preserve_price_sum_and_vector_excess_supply(self):
        # Both two and three chores catch accidental ndarray tuple-unpacking:
        # two chores silently become scalars, whereas three raise immediately.
        for n_chores in (2, 3):
            weights = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 1.0]])[:, :n_chores]
            earnings = np.array([0.8, 1.2])
            initial_prices = np.full(n_chores, earnings.sum() / n_chores)
            initial_excess = compute_excess_supply(
                initial_prices, weights, earnings, 2.0
            )
            for runner in (
                run_tatonnement,
                run_multiplicative_tatonnement,
                run_quadratic_price_tatonnement,
            ):
                with self.subTest(n_chores=n_chores, runner=runner.__name__):
                    prices, excess = runner(
                        initial_prices, weights, earnings, 2.0, 0.001, 3
                    )
                    self.assertEqual(prices.shape, (4, n_chores))
                    self.assertEqual(excess.shape, (3, n_chores))
                    self.assertTrue(np.all(prices > 0.0))
                    np.testing.assert_allclose(excess[0], initial_excess, atol=1e-12)
                    np.testing.assert_allclose(prices.sum(axis=1), earnings.sum())
                    np.testing.assert_allclose(
                        np.sum(prices[:-1] * excess, axis=1), 0.0, atol=1e-12
                    )


if __name__ == "__main__":
    unittest.main()
