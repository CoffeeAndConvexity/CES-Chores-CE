from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_1d_float_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    return arr


def _enforce_walras_law(
    prices: NDArray[np.float64],
    excess_supply: NDArray[np.float64],
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> NDArray[np.float64]:
    """Project excess supply onto the hyperplane orthogonal to prices.

    In exact arithmetic, Walras' law implies <p, z(p)> = 0. Numerical error can
    violate this by a tiny amount, so we apply the smallest correction of the form
        z_corrected = z - alpha * p
    that enforces <p, z_corrected> = 0.

    The scalar alpha is computed using means of element-wise products:
        alpha = mean(p * z) / mean(p * p)
    which is algebraically identical to alpha = <p, z> / <p, p>.

    The correction is always the minimum-norm projection needed to enforce the
    identity. This keeps each step on the Walras hyperplane even when floating-point
    accumulation makes the raw violation somewhat larger than machine epsilon.
    """
    mean_price_sq = float(np.mean(prices * prices))
    if mean_price_sq <= 0:
        raise ValueError("prices must have positive norm")

    walras_gap = float(prices @ excess_supply)
    walras_gap_mean = float(np.mean(prices * excess_supply))
    gap_scale = atol + rtol * float(
        np.linalg.norm(prices) * np.linalg.norm(excess_supply)
    )
    if abs(walras_gap) <= gap_scale:
        alpha = walras_gap_mean / mean_price_sq
        corrected = excess_supply - alpha * prices
        corrected_gap = float(prices @ corrected)
        if abs(corrected_gap) <= atol + rtol * float(
            np.linalg.norm(prices) * np.linalg.norm(corrected)
        ):
            return corrected
        return corrected - ((corrected_gap / prices.size) / mean_price_sq) * prices

    alpha = walras_gap_mean / mean_price_sq
    correction = alpha * prices
    corrected = excess_supply - correction
    corrected_gap = float(prices @ corrected)
    return corrected - ((corrected_gap / prices.size) / mean_price_sq) * prices


def compute_ces_disutility(
    allocation: ArrayLike,
    disutility_weights: ArrayLike,
    rho: float,
) -> float:
    """Compute convex CES disutility for one agent.

    Formula:
        d_i(x_i) = (sum_j (d_ij * x_ij)^rho)^(1/rho), with rho >= 1.

    Args:
        allocation: Agent allocation vector x_i of length m. Each entry x_ij must be >= 0.
        disutility_weights: Coefficients d_ij of length m. Each entry must be >= 0.
        rho: CES exponent. Must satisfy rho >= 1.

    Returns:
        The scalar disutility value for the agent.

    Raises:
        ValueError: If rho < 1, vector lengths mismatch, or inputs contain negatives.
    """
    if rho < 1:
        raise ValueError("rho must satisfy rho >= 1")

    x = _as_1d_float_array("allocation", allocation)
    d = _as_1d_float_array("disutility_weights", disutility_weights)

    if x.size != d.size:
        raise ValueError("allocation and disutility_weights must have the same length")

    if x.size == 0:
        return 0.0

    if np.any(x < 0):
        raise ValueError("allocation entries must be nonnegative")
    if np.any(d < 0):
        raise ValueError("disutility_weights entries must be nonnegative")

    total = np.sum((d * x) ** rho)
    return float(total ** (1.0 / rho))


def compute_ces_demand(
    prices: ArrayLike,
    disutility_weights: ArrayLike,
    earning_requirement: float,
    rho: float,
) -> NDArray[np.float64]:
    """Compute one agent's closed-form demand for convex CES disutility.

    Problem:
        minimize_y  d_i(y) = (sum_j (d_ij * y_ij)^rho)^(1/rho)
        subject to  <p, y> >= B_i, y >= 0

    For rho > 1 and strictly positive d_ij on goods with p_j > 0, the KKT solution is:
        y_ij = B_i * p_j^(sigma-1) * d_ij^(-sigma) / sum_k p_k^sigma * d_ik^(-sigma)
    where sigma = rho / (rho - 1).

    Args:
        prices: Price vector p of length m. Entries must be >= 0.
        disutility_weights: CES disutility coefficients d_i of length m. Entries must be >= 0.
        earning_requirement: Scalar B_i. Must be >= 0.
        rho: CES exponent. Must satisfy rho > 1.

    Returns:
        A length-m demand vector y that minimizes disutility while meeting <p, y> >= B_i.

    Raises:
        ValueError: If inputs are invalid or the earning constraint is infeasible.
    """
    if rho <= 1:
        raise ValueError("rho must satisfy rho > 1 for the closed-form demand")

    p = _as_1d_float_array("prices", prices)
    d = _as_1d_float_array("disutility_weights", disutility_weights)

    if p.size != d.size:
        raise ValueError("prices and disutility_weights must have the same length")

    if earning_requirement < 0:
        raise ValueError("earning_requirement must be nonnegative")

    m = p.size
    if m == 0:
        if earning_requirement == 0:
            return np.array([], dtype=float)
        raise ValueError(
            "infeasible: empty vectors cannot satisfy positive earning requirement"
        )

    if earning_requirement == 0:
        return np.zeros(m, dtype=float)

    if np.any(p < 0):
        raise ValueError("prices entries must be nonnegative")
    if np.any(d < 0):
        raise ValueError("disutility_weights entries must be nonnegative")

    positive_price_mask = p > 0
    if not np.any(positive_price_mask):
        raise ValueError(
            "infeasible: all prices are zero but earning_requirement is positive"
        )

    # If any positive-price good has zero disutility weight, minimum disutility is 0.
    zero_disutility_positive_price_mask = positive_price_mask & (d == 0)
    if np.any(zero_disutility_positive_price_mask):
        masked_prices = np.where(zero_disutility_positive_price_mask, p, -np.inf)
        best_j = int(np.argmax(masked_prices))
        demand = np.zeros(m, dtype=float)
        demand[best_j] = earning_requirement / p[best_j]
        return demand

    sigma = rho / (rho - 1.0)

    p_pos = p[positive_price_mask]
    d_pos = d[positive_price_mask]
    denominator = float(np.sum((p_pos**sigma) * (d_pos ** (-sigma))))

    if denominator <= 0:
        raise ValueError(
            "infeasible: could not compute a positive CES demand denominator"
        )

    demand = np.zeros(m, dtype=float)
    numerators = (p_pos ** (sigma - 1.0)) * (d_pos ** (-sigma))
    demand[positive_price_mask] = earning_requirement * (numerators / denominator)

    return demand


def compute_excess_supply(
    prices: ArrayLike,
    disutility_weights: ArrayLike,
    earning_requirements: ArrayLike,
    rho: float,
) -> NDArray[np.float64]:
    """Compute the excess supply vector for a chores Fisher market.

    Each chore j has unit supply. Excess supply is:
        z_j(p) = 1 - sum_i x_ij(p)
    where x_ij(p) is agent i's demand for chore j under prices p.

    Demand for each agent is computed in closed form (requires rho > 1):
        x_ij = B_i * p_j^(sigma-1) * d_ij^(-sigma)
                / sum_k p_k^sigma * d_ik^(-sigma)
    with sigma = rho / (rho - 1).

    Args:
        prices: Price vector p of length m. Entries must be >= 0.
        disutility_weights: Disutility coefficient matrix D of shape (n, m).
            Each entry must be > 0.
        earning_requirements: Earning requirement vector B of length n.
            Each entry must be >= 0.
        rho: CES exponent. Must satisfy rho > 1.

    Returns:
        Excess supply vector z(p) of length m.

    Raises:
        ValueError: If inputs are invalid.
    """
    if rho <= 1:
        raise ValueError("rho must satisfy rho > 1 for the closed-form demand")

    p = _as_1d_float_array("prices", prices)
    B = _as_1d_float_array("earning_requirements", earning_requirements)

    D = np.asarray(disutility_weights, dtype=float)
    if D.ndim != 2:
        raise ValueError("disutility_weights must be a 2D matrix of shape (n, m)")

    n, m = D.shape
    if p.size != m:
        raise ValueError(
            f"prices has length {p.size} but disutility_weights has {m} columns"
        )
    if B.size != n:
        raise ValueError(
            f"earning_requirements has length {B.size} but disutility_weights has {n} rows"
        )
    if np.any(p < 0):
        raise ValueError("prices entries must be nonnegative")
    if np.any(D <= 0):
        raise ValueError("disutility_weights entries must be strictly positive")
    if np.any(B < 0):
        raise ValueError("earning_requirements entries must be nonnegative")

    positive_price_mask = p > 0
    if not np.any(positive_price_mask):
        raise ValueError("infeasible: all prices are zero")

    sigma = rho / (rho - 1.0)

    p_pos = p[positive_price_mask]  # (m_pos,)
    D_pos = D[:, positive_price_mask]  # (n, m_pos)

    # denominators[i] = sum_j p_j^sigma * d_ij^(-sigma)
    denominators = (D_pos ** (-sigma)) @ (p_pos**sigma)  # (n,)

    # numerators[i, j] = p_j^(sigma-1) * d_ij^(-sigma)
    numerators = (p_pos ** (sigma - 1.0)) * (D_pos ** (-sigma))  # (n, m_pos)

    # demands[i, j] = B_i * numerators[i, j] / denominators[i]
    demands_pos = B[:, None] * numerators / denominators[:, None]  # (n, m_pos)

    # total_demand[j] = sum_i x_ij
    total_demand = np.zeros(m, dtype=float)
    total_demand[positive_price_mask] = demands_pos.sum(axis=0)

    return 1.0 - total_demand


def run_tatonnement(
    prices_init: ArrayLike,
    disutility_weights: ArrayLike,
    earning_requirements: ArrayLike,
    rho: float,
    eta: float,
    num_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run tatonnement dynamics for a chores Fisher market.

    Price update rule:
        p^{t+1} = p^t + eta * (z(p^t) - mean(z(p^t)))
    where z(p^t) is the excess supply vector. Before updating, z(p^t) is projected
    back onto the hyperplane <p^t, z> = 0 to remove small numerical Walras-law error.
    Subtracting the mean keeps the update translation-invariant (only relative prices
    matter).

    Args:
        prices_init: Initial price vector p^0 of length m. Entries must be > 0.
        disutility_weights: Disutility coefficient matrix D of shape (n, m).
            Each entry must be > 0.
        earning_requirements: Earning requirement vector B of length n.
            Each entry must be >= 0.
        rho: CES exponent. Must satisfy rho > 1.
        eta: Step size. Must be > 0.
        num_steps: Number of tatonnement iterations. Must be >= 1.

    Returns:
        prices: Price trajectory array of shape (num_steps + 1, m),
            where prices[t] is the price vector at step t.
        excess_supplies: Excess supply trajectory of shape (num_steps, m),
            where excess_supplies[t] is z(p^t).

    Raises:
        ValueError: If inputs are invalid.
    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    p0 = _as_1d_float_array("prices_init", prices_init)
    if np.any(p0 <= 0):
        raise ValueError("prices_init entries must be strictly positive")

    m = p0.size
    prices = np.empty((num_steps + 1, m), dtype=float)
    excess_supplies = np.empty((num_steps, m), dtype=float)

    prices[0] = p0
    for t in range(num_steps):
        z = compute_excess_supply(
            prices[t], disutility_weights, earning_requirements, rho
        )
        z = _enforce_walras_law(prices[t], z)
        excess_supplies[t] = z
        relative_z = z - z.mean()
        prices[t + 1] = np.maximum(prices[t] + eta * relative_z, 1e-12)

    return prices, excess_supplies


def run_multiplicative_tatonnement(
    prices_init: ArrayLike,
    disutility_weights: ArrayLike,
    earning_requirements: ArrayLike,
    rho: float,
    eta: float,
    num_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run multiplicative tatonnement dynamics for a chores Fisher market.

    Price update rule:
        u(p^t) = p^t * z(p^t)
        u_corr(p^t) = u(p^t) - mean(u(p^t))
        p^{t+1} = p^t + eta * u_corr(p^t)
    where the products are element-wise. Before forming the multiplicative update,
    z(p^t) is projected so that <p^t, z> = 0 up to numerical precision. Centering
    p^t * z(p^t) keeps the total price level stable by forcing the update to have
    zero mean.

    Args:
        prices_init: Initial price vector p^0 of length m. Entries must be > 0.
        disutility_weights: Disutility coefficient matrix D of shape (n, m).
            Each entry must be > 0.
        earning_requirements: Earning requirement vector B of length n.
            Each entry must be >= 0.
        rho: CES exponent. Must satisfy rho > 1.
        eta: Step size. Must be > 0.
        num_steps: Number of tatonnement iterations. Must be >= 1.

    Returns:
        prices: Price trajectory array of shape (num_steps + 1, m),
            where prices[t] is the price vector at step t.
        excess_supplies: Excess supply trajectory of shape (num_steps, m),
            where excess_supplies[t] is z(p^t).

    Raises:
        ValueError: If inputs are invalid.
    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    p0 = _as_1d_float_array("prices_init", prices_init)
    if np.any(p0 <= 0):
        raise ValueError("prices_init entries must be strictly positive")

    m = p0.size
    prices = np.empty((num_steps + 1, m), dtype=float)
    excess_supplies = np.empty((num_steps, m), dtype=float)

    prices[0] = p0
    for t in range(num_steps):
        z = compute_excess_supply(
            prices[t], disutility_weights, earning_requirements, rho
        )
        z = _enforce_walras_law(prices[t], z)
        excess_supplies[t] = z
        update_term = prices[t] * z
        centered_update = update_term - update_term.mean()
        prices[t + 1] = np.maximum(prices[t] + eta * centered_update, 1e-12)

    return prices, excess_supplies


def run_quadratic_price_tatonnement(
    prices_init: ArrayLike,
    disutility_weights: ArrayLike,
    earning_requirements: ArrayLike,
    rho: float,
    eta: float,
    num_steps: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run tatonnement with centered quadratic-price updates.

    Price update rule:
        u(p^t) = (p^t)^2 * z(p^t)
        u_corr(p^t) = u(p^t) - mean(u(p^t))
        p^{t+1} = p^t + eta * u_corr(p^t)
    where the powers and products are element-wise. As in the other dynamics,
    z(p^t) is first projected to satisfy <p^t, z> = 0 up to numerical precision.
    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    if num_steps < 1:
        raise ValueError("num_steps must be at least 1")

    p0 = _as_1d_float_array("prices_init", prices_init)
    if np.any(p0 <= 0):
        raise ValueError("prices_init entries must be strictly positive")

    m = p0.size
    prices = np.empty((num_steps + 1, m), dtype=float)
    excess_supplies = np.empty((num_steps, m), dtype=float)

    prices[0] = p0
    for t in range(num_steps):
        z = compute_excess_supply(
            prices[t], disutility_weights, earning_requirements, rho
        )
        z = _enforce_walras_law(prices[t], z)
        excess_supplies[t] = z
        update_term = (prices[t] ** 2) * z
        centered_update = update_term - update_term.mean()
        prices[t + 1] = np.maximum(prices[t] + eta * centered_update, 1e-12)

    return prices, excess_supplies


if __name__ == "__main__":
    # Tiny sanity check example.
    weights = np.array([13.0435, 30.4348, 30.4348, 26.087])
    allocation = np.array([1.0, 0.0, 0.5, 0.5])
    rho = 2.0
    print(compute_ces_disutility(allocation, weights, rho))

    prices = np.array([2.0, 1.0, 4.0, 3.0])
    earning_requirement = 10.0
    print(compute_ces_demand(prices, weights, earning_requirement, rho))

    # Excess supply: two agents with the same weight matrix and equal earning requirements.
    D = np.vstack([weights, weights])  # (2, 4)
    B = np.array([10.0, 10.0])
    print(compute_excess_supply(prices, D, B, rho))
