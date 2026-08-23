import numpy as np
from scipy.optimize import minimize

def geometric_mean_portfolio(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Evaluate the approximate geometric-mean objective for a portfolio."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)

    mu_p = float(x @ mu)
    var_p = float(x @ cov @ x)

    if 1.0 + mu_p <= 0:
        return -1e12

    objective_value = np.exp(
        np.log(1.0 + mu_p) - var_p / (2.0 * (1.0 + mu_p) ** 2)
    ) - 1.0
    return float(objective_value) if np.isfinite(objective_value) else -1e12


def maximize_geometric_mean(mu: np.ndarray, cov: np.ndarray, maxiter: int = 1000):
    """Maximize the approximate geometric mean with long-only weights."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = mu.size

    if cov.shape != (n, n):
        raise ValueError(f"covariance matrix must have shape ({n}, {n}), got {cov.shape}")

    result = minimize(
        lambda x: -geometric_mean_portfolio(x, mu, cov),
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x
    return {
        "weights": weights,
        "optimal_gm": geometric_mean_portfolio(weights, mu, cov),
        "mu_p": float(weights @ mu),
        "variance_p": float(weights @ cov @ weights),
        "result": result,
    }

def sharpe_ratio(x: np.ndarray, mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0) -> float:
    """Calculate a portfolio's Sharpe ratio."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    excess_return = float(x @ mu) - risk_free
    volatility = np.sqrt(max(float(x @ cov @ x), 0.0))
    return float(excess_return / volatility) if volatility else 0.0


def maximize_sharpe_ratio(
    mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0, maxiter: int = 1000
):
    """Maximize the Sharpe ratio with long-only weights."""
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = mu.size

    if cov.shape != (n, n):
        raise ValueError(f"covariance matrix must have shape ({n}, {n}), got {cov.shape}")

    result = minimize(
        lambda x: -sharpe_ratio(x, mu, cov, risk_free),
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x
    return {
        "weights": weights,
        "optimal_sharpe": sharpe_ratio(weights, mu, cov, risk_free),
        "mu_p": float(weights @ mu),
        "variance_p": float(weights @ cov @ weights),
        "risk_free": float(risk_free),
        "result": result,
    }