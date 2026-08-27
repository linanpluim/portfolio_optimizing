import numpy as np
from scipy.optimize import minimize
import pandas as pd
import yfinance as yf

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


def maximize_geometric_mean(
    returns: np.ndarray,
    periods_per_year: int = 1,
    risk_free: float = 0.0,
    maxiter: int = 1000,
):
    """Maximize the approximate geometric mean."""
    returns = np.asarray(returns, dtype=float)
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
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
    portfolio_gm = geometric_mean_portfolio(weights, mu, cov)
    portfolio_mu = float(weights @ mu)
    portfolio_variance = float(weights @ cov @ weights)
    historical_sharpe = sharpe_ratio(weights, mu, cov, risk_free)

    return {
        "weights": weights,
        "optimal_gm": portfolio_gm,
        "annual_gm": (1 + portfolio_gm) ** periods_per_year - 1,
        "mu_p": portfolio_mu,
        "annual_mu_p": portfolio_mu * periods_per_year,
        "variance_p": portfolio_variance,
        "annual_variance_p": portfolio_variance * periods_per_year,
        "historical_sharpe": historical_sharpe,
        "annual_historical_sharpe": historical_sharpe * np.sqrt(periods_per_year),
        "result": result,
    }


def sample_geometric_mean(x: np.ndarray, returns: np.ndarray) -> float:
    """Evaluate the exact geometric mean of realized portfolio returns."""
    x = np.asarray(x, dtype=float)
    returns = np.asarray(returns, dtype=float)

    growth = 1.0 + returns @ x
    if np.any(growth <= 0.0):
        return -1e12

    objective_value = np.prod(growth) ** (1.0 / returns.shape[0]) - 1.0
    return float(objective_value)


def maximize_sample_geometric_mean(
    returns: np.ndarray,
    periods_per_year: int = 1,
    risk_free: float = 0.0,
    maxiter: int = 1000,
):
    """Maximize the exact sample geometric mean."""
    returns = np.asarray(returns, dtype=float)
    observations, assets = returns.shape

    result = minimize(
        lambda x: -sample_geometric_mean(x, returns),
        np.full(assets, 1.0 / assets),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * assets,
        constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1.0},
        options={"maxiter": maxiter, "ftol": 1e-12, "disp": False},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    weights = result.x
    portfolio_returns = returns @ weights
    portfolio_gm = sample_geometric_mean(weights, returns)
    portfolio_mu = float(np.mean(portfolio_returns))
    portfolio_variance = float(np.var(portfolio_returns, ddof=1))
    historical_sharpe = sharpe_ratio(
        weights, returns.mean(axis=0), np.cov(returns, rowvar=False), risk_free
    )
    return {
        "weights": weights,
        "optimal_gm": portfolio_gm,
        "annual_gm": (1 + portfolio_gm) ** periods_per_year - 1,
        "mu_p": portfolio_mu,
        "annual_mu_p": portfolio_mu * periods_per_year,
        "variance_p": portfolio_variance,
        "annual_variance_p": portfolio_variance * periods_per_year,
        "historical_sharpe": historical_sharpe,
        "annual_historical_sharpe": historical_sharpe * np.sqrt(periods_per_year),
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
    returns: np.ndarray,
    periods_per_year: int = 1,
    risk_free: float = 0.0,
    maxiter: int = 1000,
):
    """Maximize the Sharpe ratio."""
    returns = np.asarray(returns, dtype=float)
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)

    n = mu.size
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
    portfolio_sharpe = sharpe_ratio(weights, mu, cov, risk_free)
    historical_gm = sample_geometric_mean(weights, returns)

    return {
        "weights": weights,
        "optimal_sharpe": portfolio_sharpe,
        "annual_sharpe": portfolio_sharpe * np.sqrt(periods_per_year),
        "mu_p": float(weights @ mu),
        "annual_mu_p": float(weights @ mu) * periods_per_year,
        "variance_p": float(weights @ cov @ weights),
        "annual_variance_p": float(weights @ cov @ weights) * periods_per_year,
        "historical_gm": historical_gm,
        "annual_historical_gm": (1 + historical_gm) ** periods_per_year - 1,
        "risk_free": float(risk_free),
        "result": result,
    }

def annualized_stats(returns, interval):
    """Calculate annualized statistics from periodic asset returns."""

    periods_per_year = {
        "1d": 252,
        "1wk": 52,
        "1mo": 12,
    }[interval]

    returns = returns.dropna(how="all")

    annual_mu = returns.mean() * periods_per_year
    annual_cov = returns.cov() * periods_per_year
    annual_vol = returns.std() * np.sqrt(periods_per_year)

    historical_return = (
        (1 + returns).prod()
        ** (periods_per_year / returns.notna().sum())
        - 1
    )

    stats = pd.DataFrame(
        {
            "geometric_return": historical_return,
            "arithmetic_return": annual_mu,
            "volatility": annual_vol,
        }
    )

    return stats, annual_mu, annual_cov


def download_returns(
    tickers,
    interval,
    period="max",
    start=None,
    end=None,
):
    """Download prices and return returns, mean returns, and covariance."""

    download_args = {
        "interval": interval,
        "start": start if start is not None else "1900-01-01",
        "end": end,
        "auto_adjust": True,
        "period": period,
    }
    data = yf.download(tickers, **download_args)

    timeseries = data["Close"]
    returns = timeseries.pct_change().dropna()

    return returns, returns.mean().to_numpy(), returns.cov().to_numpy()