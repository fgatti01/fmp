"""Core financial calculations - Single source of truth.

This module contains fundamental calculations used across CFA, FRM, CQF,
and Portfolio modules. All modules should import from here to avoid duplication.

Categories:
- Basic Statistics (mean, std, variance, covariance, correlation)
- Portfolio Calculations (return, variance, volatility)
- Performance Metrics (Sharpe, Sortino, max drawdown)
- Risk Metrics (VaR basic calculations)
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


# =============================================================================
# BASIC STATISTICS
# =============================================================================


def mean(data: NDArray[np.float64]) -> float:
    """Calculate arithmetic mean."""
    return float(np.mean(data))


def std(data: NDArray[np.float64], ddof: int = 1) -> float:
    """Calculate standard deviation.

    Args:
        data: Data array.
        ddof: Delta degrees of freedom (1 for sample, 0 for population).
    """
    return float(np.std(data, ddof=ddof))


def variance(data: NDArray[np.float64], ddof: int = 1) -> float:
    """Calculate variance.

    Args:
        data: Data array.
        ddof: Delta degrees of freedom (1 for sample, 0 for population).
    """
    return float(np.var(data, ddof=ddof))


def covariance(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Calculate covariance between two arrays."""
    return float(np.cov(x, y)[0, 1])


def correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Calculate Pearson correlation coefficient."""
    return float(np.corrcoef(x, y)[0, 1])


def skewness(data: NDArray[np.float64]) -> float:
    """Calculate skewness of data."""
    return float(stats.skew(data))


def kurtosis(data: NDArray[np.float64]) -> float:
    """Calculate excess kurtosis of data."""
    return float(stats.kurtosis(data))


def covariance_matrix(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate covariance matrix from returns.

    Args:
        returns: T x N array of returns (T periods, N assets).

    Returns:
        N x N covariance matrix.
    """
    return np.cov(returns, rowvar=False)


# =============================================================================
# RETURN CALCULATIONS
# =============================================================================


def simple_return(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate simple returns from prices."""
    return np.diff(prices) / prices[:-1]


def log_return(prices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate log returns from prices."""
    return np.diff(np.log(prices))


def annualize_return(
    returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Annualize returns using compound growth.

    Args:
        returns: Array of periodic returns.
        periods_per_year: Number of periods per year (252 for daily).
    """
    total_return = np.prod(1 + returns) - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year
    return float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0


def annualize_volatility(
    returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Annualize volatility by scaling standard deviation.

    Args:
        returns: Array of periodic returns.
        periods_per_year: Number of periods per year (252 for daily).
    """
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))


def geometric_mean_return(returns: NDArray[np.float64]) -> float:
    """Calculate geometric mean return."""
    return float(np.prod(1 + returns) ** (1 / len(returns)) - 1)


# =============================================================================
# PORTFOLIO CALCULATIONS
# =============================================================================


def portfolio_return(
    weights: NDArray[np.float64],
    returns: NDArray[np.float64],
) -> float:
    """Calculate portfolio return from weights and asset returns.

    Formula: R_p = w' * R

    Args:
        weights: Portfolio weights (sum to 1).
        returns: Expected returns for each asset.
    """
    return float(np.dot(weights, returns))


def portfolio_variance(
    weights: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
) -> float:
    """Calculate portfolio variance from weights and covariance matrix.

    Formula: σ²_p = w' * Σ * w

    Args:
        weights: Portfolio weights.
        cov_matrix: Covariance matrix of asset returns.
    """
    return float(np.dot(weights, np.dot(cov_matrix, weights)))


def portfolio_volatility(
    weights: NDArray[np.float64],
    cov_matrix: NDArray[np.float64],
) -> float:
    """Calculate portfolio volatility (standard deviation).

    Formula: σ_p = sqrt(w' * Σ * w)

    Args:
        weights: Portfolio weights.
        cov_matrix: Covariance matrix of asset returns.
    """
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================


def sharpe_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sharpe ratio.

    Formula: SR = (R_p - R_f) / σ_p

    Args:
        returns: Array of returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Periods per year for annualization.
    """
    ann_return = annualize_return(returns, periods_per_year)
    ann_vol = annualize_volatility(returns, periods_per_year)

    if ann_vol == 0:
        return 0.0

    return (ann_return - risk_free_rate) / ann_vol


def sortino_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Sortino ratio using downside deviation.

    Formula: Sortino = (R_p - R_f) / σ_downside

    Args:
        returns: Array of returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Periods per year for annualization.
    """
    ann_return = annualize_return(returns, periods_per_year)

    # Downside deviation
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if ann_return > risk_free_rate else 0.0

    downside_vol = float(np.std(negative_returns, ddof=1) * np.sqrt(periods_per_year))

    if downside_vol == 0:
        return 0.0

    return (ann_return - risk_free_rate) / downside_vol


def information_ratio(
    returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Calculate Information ratio.

    Formula: IR = Active Return / Tracking Error

    Args:
        returns: Portfolio returns.
        benchmark_returns: Benchmark returns.
        periods_per_year: Periods per year for annualization.
    """
    active_returns = returns - benchmark_returns

    active_return_ann = np.mean(active_returns) * periods_per_year
    tracking_err = np.std(active_returns, ddof=1) * np.sqrt(periods_per_year)

    if tracking_err == 0:
        return 0.0

    return float(active_return_ann / tracking_err)


def tracking_error(
    returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    periods_per_year: int = 252,
) -> float:
    """Calculate tracking error (annualized).

    Formula: TE = σ(R_p - R_b) * √periods_per_year

    Args:
        returns: Portfolio returns.
        benchmark_returns: Benchmark returns.
        periods_per_year: Periods per year for annualization.
    """
    active_returns = returns - benchmark_returns
    return float(np.std(active_returns, ddof=1) * np.sqrt(periods_per_year))


# =============================================================================
# DRAWDOWN CALCULATIONS
# =============================================================================


def max_drawdown(returns: NDArray[np.float64]) -> float:
    """Calculate maximum drawdown.

    Formula: MDD = max((Peak - Trough) / Peak)

    Args:
        returns: Array of returns.

    Returns:
        Maximum drawdown as a negative percentage.
    """
    wealth_index = np.cumprod(1 + returns)
    previous_peak = np.maximum.accumulate(wealth_index)
    drawdown = (wealth_index - previous_peak) / previous_peak
    return float(np.min(drawdown))


def drawdown_series(returns: NDArray[np.float64]) -> dict[str, Any]:
    """Calculate complete drawdown series and statistics.

    Args:
        returns: Array of returns.

    Returns:
        Dictionary with wealth_index, drawdown series, and statistics.
    """
    wealth_index = np.cumprod(1 + returns)
    previous_peak = np.maximum.accumulate(wealth_index)
    drawdown = (wealth_index - previous_peak) / previous_peak

    max_dd = float(np.min(drawdown))
    max_dd_idx = int(np.argmin(drawdown))

    return {
        "wealth_index": wealth_index,
        "previous_peak": previous_peak,
        "drawdown": drawdown,
        "max_drawdown": max_dd,
        "max_drawdown_end_idx": max_dd_idx,
        "current_drawdown": float(drawdown[-1]),
        "avg_drawdown": float(np.mean(drawdown[drawdown < 0])) if np.any(drawdown < 0) else 0.0,
    }


def calmar_ratio(
    returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Calmar ratio (return / max drawdown).

    Formula: Calmar = Annualized Return / |Max Drawdown|

    Args:
        returns: Array of returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Periods per year for annualization.
    """
    ann_return = annualize_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))

    if mdd == 0:
        return 0.0

    return (ann_return - risk_free_rate) / mdd


# =============================================================================
# BETA AND REGRESSION
# =============================================================================


def beta(
    returns: NDArray[np.float64],
    market_returns: NDArray[np.float64],
) -> float:
    """Calculate beta (systematic risk).

    Formula: β = Cov(R_p, R_m) / Var(R_m)

    Args:
        returns: Asset/portfolio returns.
        market_returns: Market returns.
    """
    cov = np.cov(returns, market_returns)[0, 1]
    market_var = np.var(market_returns, ddof=1)

    if market_var == 0:
        return 0.0

    return float(cov / market_var)


def alpha(
    returns: NDArray[np.float64],
    market_returns: NDArray[np.float64],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate Jensen's alpha.

    Formula: α = R_p - [R_f + β(R_m - R_f)]

    Args:
        returns: Portfolio returns.
        market_returns: Market returns.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Periods per year for annualization.
    """
    b = beta(returns, market_returns)

    rf_period = risk_free_rate / periods_per_year
    portfolio_excess = np.mean(returns) - rf_period
    market_excess = np.mean(market_returns) - rf_period

    alpha_period = portfolio_excess - b * market_excess
    return float(alpha_period * periods_per_year)


def r_squared(
    returns: NDArray[np.float64],
    market_returns: NDArray[np.float64],
) -> float:
    """Calculate R-squared (coefficient of determination).

    Args:
        returns: Portfolio returns.
        market_returns: Market returns.
    """
    corr = correlation(returns, market_returns)
    return float(corr ** 2)


# =============================================================================
# BASIC VAR CALCULATIONS
# =============================================================================


def historical_var(
    returns: NDArray[np.float64],
    confidence_level: float = 0.95,
) -> float:
    """Calculate historical Value at Risk.

    Args:
        returns: Array of returns.
        confidence_level: Confidence level (e.g., 0.95 for 95%).

    Returns:
        VaR as a positive number (potential loss).
    """
    percentile = (1 - confidence_level) * 100
    return float(-np.percentile(returns, percentile))


def parametric_var(
    returns: NDArray[np.float64],
    confidence_level: float = 0.95,
) -> float:
    """Calculate parametric (Gaussian) Value at Risk.

    Formula: VaR = -(μ + z * σ)

    Args:
        returns: Array of returns.
        confidence_level: Confidence level (e.g., 0.95 for 95%).

    Returns:
        VaR as a positive number (potential loss).
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z = stats.norm.ppf(1 - confidence_level)
    return float(-(mu + z * sigma))


def expected_shortfall(
    returns: NDArray[np.float64],
    confidence_level: float = 0.95,
) -> float:
    """Calculate Expected Shortfall (CVaR).

    The average loss beyond VaR threshold.

    Args:
        returns: Array of returns.
        confidence_level: Confidence level (e.g., 0.95 for 95%).

    Returns:
        ES as a positive number (average loss beyond VaR).
    """
    var = historical_var(returns, confidence_level)
    losses_beyond_var = returns[returns <= -var]

    if len(losses_beyond_var) == 0:
        return var

    return float(-np.mean(losses_beyond_var))


# =============================================================================
# SEMIDEVIATION
# =============================================================================


def semideviation(returns: NDArray[np.float64]) -> float:
    """Calculate semideviation (downside standard deviation).

    Measures volatility of returns below the mean.

    Args:
        returns: Array of returns.
    """
    mean_return = np.mean(returns)
    negative_returns = returns[returns < mean_return]

    if len(negative_returns) == 0:
        return 0.0

    return float(np.std(negative_returns, ddof=1))


def downside_deviation(
    returns: NDArray[np.float64],
    threshold: float = 0.0,
) -> float:
    """Calculate downside deviation relative to threshold.

    Formula: sqrt(E[min(r - threshold, 0)²])

    Args:
        returns: Array of returns.
        threshold: Minimum acceptable return (MAR).
    """
    below_threshold = returns - threshold
    below_threshold = below_threshold[below_threshold < 0]

    if len(below_threshold) == 0:
        return 0.0

    return float(np.sqrt(np.mean(below_threshold ** 2)))
