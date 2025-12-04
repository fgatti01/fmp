"""Portfolio Management metrics and optimization.

This module implements portfolio management frameworks based on the
Norte Asset Management Quant Finance Master Guide:

1. Strategic Asset Allocation (Mean-Variance Optimization)
2. Tactical Asset Allocation
3. Black-Litterman Model
4. Risk Parity
5. Constrained Optimization
6. Monte Carlo Planning
7. Glide Paths (Target Date)
8. Factor-Based Allocation
9. Minimum Variance Portfolio
10. Maximum Diversification Portfolio
11. Equal Risk Contribution
12. Kelly Criterion

Note: Basic portfolio and statistical calculations use metrics.core
for consistency across all modules.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize

from fmp_analytics.metrics import core


@dataclass
class PortfolioResult:
    """Portfolio optimization result."""

    weights: NDArray[np.float64]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    diversification_ratio: float


@dataclass
class RiskParityResult:
    """Risk parity portfolio result."""

    weights: NDArray[np.float64]
    risk_contributions: NDArray[np.float64]
    portfolio_volatility: float
    max_weight: float
    min_weight: float


@dataclass
class BlackLittermanResult:
    """Black-Litterman model result."""

    posterior_returns: NDArray[np.float64]
    posterior_covariance: NDArray[np.float64]
    optimal_weights: NDArray[np.float64]
    expected_return: float
    volatility: float


class PortfolioManagement:
    """Portfolio management and optimization framework."""

    # ==================== MEAN-VARIANCE OPTIMIZATION ====================

    @staticmethod
    def mean_variance_optimization(
        expected_returns: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        target_return: float | None = None,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> PortfolioResult:
        """Markowitz Mean-Variance Optimization.

        Maximizes Sharpe Ratio or targets a specific return.

        Formula:
            max (w'μ - Rf) / sqrt(w'Σw)

        Args:
            expected_returns: Expected returns for each asset.
            cov_matrix: Covariance matrix of returns.
            risk_free_rate: Risk-free rate.
            target_return: Target return (if None, maximizes Sharpe).
            max_weight: Maximum weight per asset.
            min_weight: Minimum weight per asset.

        Returns:
            PortfolioResult with optimal weights.
        """
        n_assets = len(expected_returns)

        def neg_sharpe(weights: NDArray[np.float64]) -> float:
            ret = np.dot(weights, expected_returns)
            vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if vol == 0:
                return 0
            return -(ret - risk_free_rate) / vol

        def portfolio_variance(weights: NDArray[np.float64]) -> float:
            return np.dot(weights, np.dot(cov_matrix, weights))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        if target_return is not None:
            constraints.append(
                {"type": "eq", "fun": lambda w: np.dot(w, expected_returns) - target_return}
            )
            objective = portfolio_variance
        else:
            objective = neg_sharpe

        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        optimal_weights = result.x
        ret = float(np.dot(optimal_weights, expected_returns))
        vol = float(np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

        # Diversification ratio
        weighted_vols = np.sqrt(np.diag(cov_matrix)) * optimal_weights
        diversification = np.sum(weighted_vols) / vol if vol > 0 else 1

        return PortfolioResult(
            weights=optimal_weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=float(sharpe),
            diversification_ratio=float(diversification),
        )

    @staticmethod
    def efficient_frontier(
        expected_returns: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        n_points: int = 50,
        risk_free_rate: float = 0.0,
    ) -> dict[str, NDArray[np.float64]]:
        """Generate the efficient frontier.

        Args:
            expected_returns: Expected returns for each asset.
            cov_matrix: Covariance matrix.
            n_points: Number of points on the frontier.
            risk_free_rate: Risk-free rate.

        Returns:
            Dictionary with returns, volatilities, and Sharpe ratios.
        """
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_returns = []
        frontier_vols = []
        frontier_sharpes = []

        for target in target_returns:
            try:
                result = PortfolioManagement.mean_variance_optimization(
                    expected_returns, cov_matrix, risk_free_rate, target_return=target
                )
                frontier_returns.append(result.expected_return)
                frontier_vols.append(result.volatility)
                frontier_sharpes.append(result.sharpe_ratio)
            except Exception:
                continue

        return {
            "returns": np.array(frontier_returns),
            "volatilities": np.array(frontier_vols),
            "sharpe_ratios": np.array(frontier_sharpes),
        }

    # ==================== MINIMUM VARIANCE PORTFOLIO ====================

    @staticmethod
    def minimum_variance_portfolio(
        cov_matrix: NDArray[np.float64],
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> NDArray[np.float64]:
        """Calculate minimum variance portfolio weights.

        Formula:
            min w'Σw s.t. Σw = 1

        Args:
            cov_matrix: Covariance matrix.
            max_weight: Maximum weight per asset.
            min_weight: Minimum weight per asset.

        Returns:
            Optimal weights.
        """
        n_assets = cov_matrix.shape[0]

        def portfolio_variance(weights: NDArray[np.float64]) -> float:
            return np.dot(weights, np.dot(cov_matrix, weights))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(
            portfolio_variance, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        return result.x

    # ==================== RISK PARITY ====================

    @staticmethod
    def risk_parity_portfolio(
        cov_matrix: NDArray[np.float64],
        risk_budgets: NDArray[np.float64] | None = None,
    ) -> RiskParityResult:
        """Calculate risk parity portfolio weights.

        Equalizes risk contribution from each asset.

        Formula:
            RC_i = w_i * (Σw)_i / sqrt(w'Σw)

        Args:
            cov_matrix: Covariance matrix.
            risk_budgets: Target risk budget per asset (default: equal).

        Returns:
            RiskParityResult with weights and contributions.
        """
        n_assets = cov_matrix.shape[0]

        if risk_budgets is None:
            risk_budgets = np.ones(n_assets) / n_assets

        def risk_contribution(weights: NDArray[np.float64]) -> NDArray[np.float64]:
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / port_vol
            return weights * marginal_contrib

        def objective(weights: NDArray[np.float64]) -> float:
            contrib = risk_contribution(weights)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            target_contrib = risk_budgets * port_vol
            return float(np.sum((contrib - target_contrib) ** 2))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0.001, 1) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        risk_contribs = risk_contribution(optimal_weights)
        port_vol = float(np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))))

        return RiskParityResult(
            weights=optimal_weights,
            risk_contributions=risk_contribs,
            portfolio_volatility=port_vol,
            max_weight=float(optimal_weights.max()),
            min_weight=float(optimal_weights.min()),
        )

    # ==================== BLACK-LITTERMAN ====================

    @staticmethod
    def black_litterman(
        market_weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        P: NDArray[np.float64],
        Q: NDArray[np.float64],
        tau: float = 0.025,
        risk_aversion: float = 2.5,
        omega: NDArray[np.float64] | None = None,
    ) -> BlackLittermanResult:
        """Black-Litterman asset allocation model.

        Combines market equilibrium returns with investor views.

        Formula:
            Posterior = (τΣ)^(-1)Π + P'Ω^(-1)Q
            E[R] = [(τΣ)^(-1) + P'Ω^(-1)P]^(-1) * Posterior

        Args:
            market_weights: Market capitalization weights.
            cov_matrix: Covariance matrix.
            P: View matrix (K views x N assets).
            Q: View returns (K x 1).
            tau: Uncertainty in equilibrium (typically 0.025-0.05).
            risk_aversion: Risk aversion coefficient.
            omega: View uncertainty matrix (if None, uses P * tau * Σ * P').

        Returns:
            BlackLittermanResult with posterior returns and weights.
        """
        # Equilibrium returns (reverse optimization)
        Pi = risk_aversion * np.dot(cov_matrix, market_weights)

        # Uncertainty in prior
        tau_sigma = tau * cov_matrix

        # View uncertainty
        if omega is None:
            omega = np.diag(np.diag(P @ tau_sigma @ P.T))

        # Black-Litterman formula
        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega)

        # Posterior precision
        posterior_precision = inv_tau_sigma + P.T @ inv_omega @ P

        # Posterior covariance
        posterior_cov = np.linalg.inv(posterior_precision)

        # Posterior mean
        posterior_returns = posterior_cov @ (inv_tau_sigma @ Pi + P.T @ inv_omega @ Q)

        # Optimal weights (unconstrained)
        optimal_weights = (1 / risk_aversion) * np.linalg.inv(cov_matrix) @ posterior_returns

        # Normalize weights
        optimal_weights = optimal_weights / np.sum(np.abs(optimal_weights))
        optimal_weights = np.maximum(optimal_weights, 0)  # Long only
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        ret = float(np.dot(optimal_weights, posterior_returns))
        vol = float(np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))))

        return BlackLittermanResult(
            posterior_returns=posterior_returns,
            posterior_covariance=posterior_cov,
            optimal_weights=optimal_weights,
            expected_return=ret,
            volatility=vol,
        )

    # ==================== TACTICAL ASSET ALLOCATION ====================

    @staticmethod
    def tactical_tilt(
        policy_weights: NDArray[np.float64],
        signals: NDArray[np.float64],
        max_tilt: float = 0.10,
    ) -> NDArray[np.float64]:
        """Apply tactical tilts to strategic allocation.

        Args:
            policy_weights: Strategic asset allocation weights.
            signals: Signal strength (-1 to 1) for each asset.
            max_tilt: Maximum tilt from policy weight.

        Returns:
            Tactical weights.
        """
        # Apply tilts
        adjustments = signals * max_tilt
        tactical_weights = policy_weights * (1 + adjustments)

        # Normalize
        tactical_weights = tactical_weights / np.sum(tactical_weights)

        return tactical_weights

    # ==================== MONTE CARLO SIMULATION ====================

    @staticmethod
    def monte_carlo_wealth_simulation(
        initial_wealth: float,
        expected_return: float,
        volatility: float,
        annual_contribution: float = 0,
        annual_withdrawal: float = 0,
        years: int = 30,
        n_simulations: int = 10000,
    ) -> dict[str, Any]:
        """Monte Carlo simulation for wealth projection.

        Args:
            initial_wealth: Starting wealth.
            expected_return: Expected annual return.
            volatility: Annual volatility.
            annual_contribution: Annual contribution (positive).
            annual_withdrawal: Annual withdrawal (positive).
            years: Investment horizon.
            n_simulations: Number of simulations.

        Returns:
            Dictionary with simulation results.
        """
        final_values = np.zeros(n_simulations)

        for sim in range(n_simulations):
            wealth = initial_wealth

            for _ in range(years):
                # Random return
                annual_return = np.random.normal(expected_return, volatility)

                # Update wealth
                wealth = wealth * (1 + annual_return) + annual_contribution - annual_withdrawal

                if wealth <= 0:
                    break

            final_values[sim] = max(0, wealth)

        # Success probability (wealth > 0)
        prob_success = np.sum(final_values > 0) / n_simulations

        return {
            "initial_wealth": initial_wealth,
            "years": years,
            "prob_success": float(prob_success),
            "expected_final": float(np.mean(final_values)),
            "median_final": float(np.median(final_values)),
            "p10": float(np.percentile(final_values, 10)),
            "p25": float(np.percentile(final_values, 25)),
            "p50": float(np.percentile(final_values, 50)),
            "p75": float(np.percentile(final_values, 75)),
            "p90": float(np.percentile(final_values, 90)),
            "std": float(np.std(final_values)),
        }

    # ==================== GLIDE PATH ====================

    @staticmethod
    def target_date_glide_path(
        current_age: int,
        retirement_age: int = 65,
        initial_equity: float = 0.90,
        final_equity: float = 0.40,
    ) -> dict[str, Any]:
        """Calculate target-date fund glide path.

        Args:
            current_age: Current age.
            retirement_age: Target retirement age.
            initial_equity: Equity allocation at start.
            final_equity: Equity allocation at retirement.

        Returns:
            Dictionary with glide path allocations.
        """
        years_to_retirement = max(0, retirement_age - current_age)

        # Linear glide path
        if years_to_retirement > 0:
            annual_reduction = (initial_equity - final_equity) / years_to_retirement
            current_equity = initial_equity - annual_reduction * (retirement_age - current_age - years_to_retirement)
            current_equity = max(final_equity, min(initial_equity, current_equity))
        else:
            current_equity = final_equity

        return {
            "current_equity": float(current_equity),
            "current_fixed_income": float(1 - current_equity),
            "years_to_retirement": years_to_retirement,
            "annual_equity_reduction": float((initial_equity - final_equity) / max(1, retirement_age - 25)),
        }

    # ==================== KELLY CRITERION ====================

    @staticmethod
    def kelly_criterion(
        expected_returns: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        fraction: float = 0.5,
    ) -> NDArray[np.float64]:
        """Calculate Kelly criterion optimal weights.

        Formula:
            f* = Σ^(-1) * (μ - Rf) / γ

        Args:
            expected_returns: Expected returns.
            cov_matrix: Covariance matrix.
            risk_free_rate: Risk-free rate.
            fraction: Kelly fraction (0.5 = half-Kelly).

        Returns:
            Kelly optimal weights.
        """
        excess_returns = expected_returns - risk_free_rate

        # Full Kelly
        inv_cov = np.linalg.inv(cov_matrix)
        full_kelly = inv_cov @ excess_returns

        # Apply fraction
        kelly_weights = fraction * full_kelly

        # Normalize to sum to 1
        kelly_weights = kelly_weights / np.sum(np.abs(kelly_weights))

        return kelly_weights

    # ==================== MAXIMUM DIVERSIFICATION ====================

    @staticmethod
    def maximum_diversification_portfolio(
        cov_matrix: NDArray[np.float64],
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> NDArray[np.float64]:
        """Calculate maximum diversification portfolio.

        Maximizes the diversification ratio.

        Formula:
            max (w'σ) / sqrt(w'Σw)

        Args:
            cov_matrix: Covariance matrix.
            max_weight: Maximum weight per asset.
            min_weight: Minimum weight per asset.

        Returns:
            Optimal weights.
        """
        n_assets = cov_matrix.shape[0]
        asset_vols = np.sqrt(np.diag(cov_matrix))

        def neg_diversification_ratio(weights: NDArray[np.float64]) -> float:
            weighted_vols = np.dot(weights, asset_vols)
            port_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if port_vol == 0:
                return 0
            return -weighted_vols / port_vol

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(
            neg_diversification_ratio, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        return result.x

    # ==================== ASSET LIABILITY MATCHING ====================

    @staticmethod
    def liability_driven_investment(
        liability_pv: float,
        liability_duration: float,
        asset_durations: NDArray[np.float64],
        asset_convexities: NDArray[np.float64],
        expected_returns: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Liability-Driven Investment (LDI) analysis.

        Args:
            liability_pv: Present value of liabilities.
            liability_duration: Duration of liabilities.
            asset_durations: Duration of each asset.
            asset_convexities: Convexity of each asset.
            expected_returns: Expected return of each asset.

        Returns:
            Dictionary with LDI metrics and suggested weights.
        """
        n_assets = len(expected_returns)

        # Duration matching constraint
        def duration_match_objective(weights: NDArray[np.float64]) -> float:
            portfolio_duration = np.dot(weights, asset_durations)
            return (portfolio_duration - liability_duration) ** 2

        # Maximize return subject to duration match
        def neg_return(weights: NDArray[np.float64]) -> float:
            return -np.dot(weights, expected_returns)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: np.dot(w, asset_durations) - liability_duration},
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(neg_return, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        optimal_weights = result.x
        portfolio_duration = float(np.dot(optimal_weights, asset_durations))
        portfolio_convexity = float(np.dot(optimal_weights, asset_convexities))

        return {
            "optimal_weights": optimal_weights,
            "portfolio_duration": portfolio_duration,
            "portfolio_convexity": portfolio_convexity,
            "liability_duration": liability_duration,
            "duration_gap": float(portfolio_duration - liability_duration),
            "expected_return": float(np.dot(optimal_weights, expected_returns)),
        }

    # ==================== FACTOR ALLOCATION ====================

    @staticmethod
    def factor_based_allocation(
        factor_exposures: NDArray[np.float64],
        target_exposures: NDArray[np.float64],
        expected_returns: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Factor-based portfolio allocation.

        Target specific factor exposures.

        Args:
            factor_exposures: Factor exposure matrix (assets x factors).
            target_exposures: Target factor exposures.
            expected_returns: Expected returns.
            cov_matrix: Covariance matrix.

        Returns:
            Optimal weights.
        """
        n_assets = len(expected_returns)
        n_factors = len(target_exposures)

        def objective(weights: NDArray[np.float64]) -> float:
            # Minimize tracking error to factor targets
            actual_exposures = factor_exposures.T @ weights
            tracking = np.sum((actual_exposures - target_exposures) ** 2)
            # Plus portfolio variance
            variance = np.dot(weights, np.dot(cov_matrix, weights))
            return tracking + 0.1 * variance

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1 / n_assets] * n_assets)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        return result.x

    # ==================== REBALANCING ====================

    @staticmethod
    def rebalancing_analysis(
        current_weights: NDArray[np.float64],
        target_weights: NDArray[np.float64],
        portfolio_value: float,
        transaction_cost: float = 0.001,
        threshold: float = 0.05,
    ) -> dict[str, Any]:
        """Analyze rebalancing needs and costs.

        Args:
            current_weights: Current portfolio weights.
            target_weights: Target portfolio weights.
            portfolio_value: Total portfolio value.
            transaction_cost: Transaction cost as fraction.
            threshold: Rebalancing threshold.

        Returns:
            Dictionary with rebalancing analysis.
        """
        drift = current_weights - target_weights
        abs_drift = np.abs(drift)
        max_drift = float(np.max(abs_drift))

        # Trades needed
        trades = target_weights - current_weights
        trade_value = np.abs(trades) * portfolio_value
        total_trade_value = float(np.sum(trade_value))

        # Transaction costs
        total_cost = total_trade_value * transaction_cost

        # Should rebalance?
        should_rebalance = max_drift > threshold

        return {
            "drift": drift,
            "max_drift": max_drift,
            "trades_needed": trades,
            "trade_values": trade_value,
            "total_trade_value": total_trade_value,
            "transaction_cost": float(total_cost),
            "should_rebalance": should_rebalance,
            "threshold": threshold,
        }

    # ==================== EDHEC COVARIANCE ESTIMATION ====================

    # Delegate to core for basic covariance calculation
    sample_covariance = staticmethod(core.covariance_matrix)

    @staticmethod
    def constant_correlation_covariance(
        returns: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Elton-Gruber Constant Correlation Model for covariance estimation.

        Assumes all pairwise correlations are equal to the average correlation.
        This provides a more stable covariance estimate than sample covariance.

        Formula:
            ρ_avg = average of all pairwise correlations
            Σ_cc[i,j] = σ_i × σ_j × ρ_avg for i ≠ j
            Σ_cc[i,i] = σ_i²

        Args:
            returns: T x N array of returns (T periods, N assets).

        Returns:
            Dictionary with:
                - covariance: Constant correlation covariance matrix
                - correlation: Constant correlation correlation matrix
                - average_correlation: The average pairwise correlation used
                - volatilities: Individual asset volatilities
        """
        # Sample covariance and correlation
        sample_cov = np.cov(returns, rowvar=False)
        volatilities = np.sqrt(np.diag(sample_cov))

        # Correlation matrix
        n = len(volatilities)
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_outer = np.outer(volatilities, volatilities)
            corr_matrix = np.where(vol_outer != 0, sample_cov / vol_outer, 0)

        # Average correlation (excluding diagonal)
        mask = ~np.eye(n, dtype=bool)
        avg_corr = np.mean(corr_matrix[mask])

        # Build constant correlation matrix
        cc_corr = np.full((n, n), avg_corr)
        np.fill_diagonal(cc_corr, 1.0)

        # Build covariance matrix
        cc_cov = cc_corr * np.outer(volatilities, volatilities)

        return {
            "covariance": cc_cov,
            "correlation": cc_corr,
            "average_correlation": float(avg_corr),
            "volatilities": volatilities,
        }

    @staticmethod
    def shrinkage_covariance(
        returns: NDArray[np.float64],
        shrinkage_factor: float = 0.5,
        target: str = "constant_correlation",
    ) -> dict[str, Any]:
        """Shrinkage covariance estimator.

        Blends the sample covariance with a structured target (prior) using
        a shrinkage factor. This reduces estimation error, especially with
        few observations.

        Formula:
            Σ_shrunk = δ × Σ_target + (1 - δ) × Σ_sample

        Where δ is the shrinkage factor.

        Args:
            returns: T x N array of returns.
            shrinkage_factor: Blend factor (0 = sample cov, 1 = target).
            target: Target structure - "constant_correlation" or "identity".

        Returns:
            Dictionary with:
                - covariance: Shrunk covariance matrix
                - sample_covariance: Original sample covariance
                - target_covariance: Target covariance matrix
                - shrinkage_factor: Factor used
        """
        sample_cov = np.cov(returns, rowvar=False)

        if target == "constant_correlation":
            target_result = PortfolioManagement.constant_correlation_covariance(returns)
            target_cov = target_result["covariance"]
        elif target == "identity":
            # Scaled identity matrix
            avg_var = np.mean(np.diag(sample_cov))
            n = sample_cov.shape[0]
            target_cov = np.eye(n) * avg_var
        else:
            # Default to scaled identity
            avg_var = np.mean(np.diag(sample_cov))
            n = sample_cov.shape[0]
            target_cov = np.eye(n) * avg_var

        # Shrink
        shrunk_cov = shrinkage_factor * target_cov + (1 - shrinkage_factor) * sample_cov

        return {
            "covariance": shrunk_cov,
            "sample_covariance": sample_cov,
            "target_covariance": target_cov,
            "shrinkage_factor": shrinkage_factor,
        }

    @staticmethod
    def ledoit_wolf_shrinkage(
        returns: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Ledoit-Wolf optimal shrinkage estimator.

        Automatically determines the optimal shrinkage intensity by
        minimizing the expected loss.

        Based on: Ledoit & Wolf (2004) "Honey, I Shrunk the Sample Covariance Matrix"

        Args:
            returns: T x N array of returns.

        Returns:
            Dictionary with:
                - covariance: Optimally shrunk covariance matrix
                - shrinkage_intensity: Optimal shrinkage factor
                - sample_covariance: Original sample covariance
        """
        t, n = returns.shape

        # Center the returns
        returns_centered = returns - np.mean(returns, axis=0)

        # Sample covariance
        sample_cov = np.dot(returns_centered.T, returns_centered) / t

        # Target: scaled identity
        mu = np.trace(sample_cov) / n
        target = mu * np.eye(n)

        # Compute optimal shrinkage intensity
        # Frobenius norms
        delta = sample_cov - target
        delta_sq = np.sum(delta**2)

        # Estimate asymptotic variance
        y = returns_centered**2
        phi_mat = np.dot(y.T, y) / t - sample_cov**2
        phi = np.sum(phi_mat)

        # Estimate gamma (cross-product term)
        gamma = delta_sq

        # Optimal shrinkage
        kappa = (phi - gamma) / t
        shrinkage = max(0, min(1, kappa / delta_sq)) if delta_sq > 0 else 0

        # Shrunk covariance
        shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov

        return {
            "covariance": shrunk_cov,
            "shrinkage_intensity": float(shrinkage),
            "sample_covariance": sample_cov,
            "target_covariance": target,
        }

    # ==================== RISK CONTRIBUTION ANALYSIS ====================

    @staticmethod
    def marginal_risk_contribution(
        weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Calculate marginal and component risk contributions.

        Marginal Risk Contribution (MRC): ∂σₚ/∂wᵢ
        Component Risk Contribution (CRC): wᵢ × MRC_i
        Percentage Risk Contribution: CRC_i / σₚ

        Args:
            weights: Portfolio weights.
            cov_matrix: Covariance matrix.

        Returns:
            Dictionary with:
                - marginal_risk: Marginal risk contribution of each asset
                - component_risk: Component risk contribution
                - percentage_risk: Percentage of total risk
                - portfolio_volatility: Total portfolio volatility
        """
        # Portfolio volatility
        port_var = np.dot(weights, np.dot(cov_matrix, weights))
        port_vol = np.sqrt(port_var)

        # Marginal risk contribution
        mrc = np.dot(cov_matrix, weights) / port_vol

        # Component risk contribution
        crc = weights * mrc

        # Percentage risk contribution
        prc = crc / port_vol

        return {
            "marginal_risk": mrc,
            "component_risk": crc,
            "percentage_risk": prc,
            "portfolio_volatility": float(port_vol),
            "sum_crc": float(np.sum(crc)),  # Should equal port_vol
        }

    @staticmethod
    def target_risk_contribution(
        cov_matrix: NDArray[np.float64],
        target_contributions: NDArray[np.float64],
        max_weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> dict[str, Any]:
        """Find portfolio weights to match target risk contributions.

        Useful for custom risk budgeting beyond equal risk contribution.

        Args:
            cov_matrix: Covariance matrix.
            target_contributions: Target percentage risk contributions (sum to 1).
            max_weight: Maximum weight per asset.
            min_weight: Minimum weight per asset.

        Returns:
            Dictionary with:
                - weights: Optimal weights
                - achieved_contributions: Actual risk contributions
                - error: Mean squared error vs target
        """
        n = len(cov_matrix)
        target_contributions = np.array(target_contributions)
        target_contributions = target_contributions / target_contributions.sum()

        def objective(weights):
            port_var = np.dot(weights, np.dot(cov_matrix, weights))
            port_vol = np.sqrt(port_var)

            if port_vol < 1e-10:
                return 1e10

            # Risk contributions
            mrc = np.dot(cov_matrix, weights) / port_vol
            crc = weights * mrc
            prc = crc / port_vol

            # Minimize squared error from target
            return np.sum((prc - target_contributions) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((min_weight, max_weight) for _ in range(n))
        x0 = np.array([1 / n] * n)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        # Calculate achieved contributions
        opt_weights = result.x
        port_var = np.dot(opt_weights, np.dot(cov_matrix, opt_weights))
        port_vol = np.sqrt(port_var)
        mrc = np.dot(cov_matrix, opt_weights) / port_vol
        crc = opt_weights * mrc
        prc = crc / port_vol

        return {
            "weights": opt_weights,
            "achieved_contributions": prc,
            "target_contributions": target_contributions,
            "error": float(result.fun),
            "portfolio_volatility": float(port_vol),
        }

    # ==================== STYLE & FACTOR ANALYSIS ====================

    @staticmethod
    def style_analysis(
        fund_returns: NDArray[np.float64],
        benchmark_returns: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Sharpe Style Analysis (Returns-Based).

        Estimates the portfolio's exposure to benchmark factors/indices
        using constrained regression (weights sum to 1, non-negative).

        Args:
            fund_returns: Returns of the fund to analyze.
            benchmark_returns: T x N array of benchmark returns.

        Returns:
            Dictionary with:
                - style_weights: Estimated exposure to each benchmark
                - r_squared: R-squared of the fit
                - tracking_error: Residual volatility
                - selection_return: Alpha (annualized)
        """
        n_benchmarks = benchmark_returns.shape[1]

        def objective(weights):
            replicated = np.dot(benchmark_returns, weights)
            residuals = fund_returns - replicated
            return np.sum(residuals**2)

        # Weights sum to 1, non-negative
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 1) for _ in range(n_benchmarks))
        x0 = np.array([1 / n_benchmarks] * n_benchmarks)

        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        opt_weights = result.x

        # Calculate metrics
        replicated = np.dot(benchmark_returns, opt_weights)
        residuals = fund_returns - replicated

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((fund_returns - np.mean(fund_returns))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Tracking error (annualized)
        tracking_error = np.std(residuals) * np.sqrt(252)

        # Selection return (alpha)
        selection_return = np.mean(residuals) * 252

        return {
            "style_weights": opt_weights,
            "r_squared": float(r_squared),
            "tracking_error": float(tracking_error),
            "selection_return": float(selection_return),
            "residuals": residuals,
        }

    @staticmethod
    def fama_french_analysis(
        returns: NDArray[np.float64],
        market_excess: NDArray[np.float64],
        smb: NDArray[np.float64],
        hml: NDArray[np.float64],
        risk_free: NDArray[np.float64] | None = None,
    ) -> dict[str, Any]:
        """Fama-French 3-Factor Model Analysis.

        Regresses portfolio returns on Market, Size (SMB), and Value (HML) factors.

        Model: R_p - R_f = α + β_mkt(R_m - R_f) + β_smb(SMB) + β_hml(HML) + ε

        Args:
            returns: Portfolio returns.
            market_excess: Market excess returns (R_m - R_f).
            smb: Small Minus Big factor returns.
            hml: High Minus Low (value) factor returns.
            risk_free: Risk-free rate (optional, for calculating excess returns).

        Returns:
            Dictionary with:
                - alpha: Annualized alpha (Jensen's alpha)
                - beta_market: Market beta
                - beta_smb: Size factor loading
                - beta_hml: Value factor loading
                - r_squared: Explanatory power
                - t_statistics: t-stats for each coefficient
        """
        # Calculate excess returns if risk_free provided
        if risk_free is not None:
            excess_returns = returns - risk_free
        else:
            excess_returns = returns

        # Build factor matrix
        X = np.column_stack([np.ones(len(returns)), market_excess, smb, hml])
        y = excess_returns

        # OLS regression
        try:
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.zeros(4)
            residuals = y

        alpha = beta[0]
        beta_mkt = beta[1]
        beta_smb = beta[2]
        beta_hml = beta[3]

        # R-squared
        y_pred = np.dot(X, beta)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # T-statistics (simplified)
        n = len(y)
        k = 4  # number of parameters
        mse = ss_res / (n - k) if n > k else ss_res
        var_beta = mse * np.linalg.inv(np.dot(X.T, X)).diagonal()
        t_stats = beta / np.sqrt(np.maximum(var_beta, 1e-10))

        return {
            "alpha": float(alpha * 252),  # Annualized
            "alpha_daily": float(alpha),
            "beta_market": float(beta_mkt),
            "beta_smb": float(beta_smb),
            "beta_hml": float(beta_hml),
            "r_squared": float(r_squared),
            "t_statistics": {
                "alpha": float(t_stats[0]),
                "market": float(t_stats[1]),
                "smb": float(t_stats[2]),
                "hml": float(t_stats[3]),
            },
            "interpretation": {
                "market_exposure": "High" if abs(beta_mkt) > 1.2 else "Moderate" if abs(beta_mkt) > 0.8 else "Low",
                "size_tilt": "Small-cap" if beta_smb > 0.2 else "Large-cap" if beta_smb < -0.2 else "Neutral",
                "value_tilt": "Value" if beta_hml > 0.2 else "Growth" if beta_hml < -0.2 else "Neutral",
            },
        }

    @staticmethod
    def tracking_error_analysis(
        portfolio_returns: NDArray[np.float64],
        benchmark_returns: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Comprehensive tracking error analysis.

        Args:
            portfolio_returns: Portfolio returns.
            benchmark_returns: Benchmark returns.

        Returns:
            Dictionary with:
                - tracking_error: Annualized tracking error
                - tracking_error_daily: Daily tracking error
                - active_return: Annualized active return (alpha)
                - information_ratio: Active return / Tracking error
                - active_share_estimate: Estimated active share from tracking error
                - correlation: Correlation with benchmark
        """
        active_returns = portfolio_returns - benchmark_returns

        # Tracking error
        te_daily = np.std(active_returns)
        te_annual = te_daily * np.sqrt(252)

        # Active return
        active_return_daily = np.mean(active_returns)
        active_return_annual = active_return_daily * 252

        # Information ratio
        ir = active_return_annual / te_annual if te_annual > 0 else 0

        # Correlation
        corr = np.corrcoef(portfolio_returns, benchmark_returns)[0, 1]

        # Estimate active share from tracking error (rough approximation)
        # Based on empirical relationship: TE ≈ 0.5 × Active_Share for equity funds
        active_share_estimate = min(1.0, te_annual / 0.5 * 0.2)

        return {
            "tracking_error": float(te_annual),
            "tracking_error_daily": float(te_daily),
            "active_return": float(active_return_annual),
            "active_return_daily": float(active_return_daily),
            "information_ratio": float(ir),
            "active_share_estimate": float(active_share_estimate),
            "correlation": float(corr),
            "interpretation": (
                "Index-like" if te_annual < 0.02
                else "Closet indexer" if te_annual < 0.04
                else "Active manager" if te_annual < 0.08
                else "Highly active"
            ),
        }

    # ==================== BACKTESTING FRAMEWORK ====================

    @staticmethod
    def rolling_backtest(
        returns: NDArray[np.float64],
        estimation_window: int = 252,
        rebalance_frequency: int = 21,
        strategy: str = "gmv",
        **kwargs,
    ) -> dict[str, Any]:
        """Rolling-window portfolio backtesting framework.

        Simulates a portfolio strategy with periodic rebalancing using
        historical data for estimation.

        Args:
            returns: T x N array of asset returns.
            estimation_window: Lookback period for parameter estimation.
            rebalance_frequency: Periods between rebalances.
            strategy: Portfolio strategy - "gmv", "equal_weight", "risk_parity",
                     "max_sharpe", "inverse_vol".
            **kwargs: Additional arguments for the strategy.

        Returns:
            Dictionary with:
                - portfolio_returns: Backtest returns
                - cumulative_return: Cumulative wealth
                - weights_history: Portfolio weights over time
                - turnover: Average turnover
                - metrics: Performance metrics
        """
        t, n = returns.shape
        start_idx = estimation_window

        portfolio_returns = []
        weights_history = []
        current_weights = np.array([1 / n] * n)

        for i in range(start_idx, t):
            # Rebalance at specified frequency
            if (i - start_idx) % rebalance_frequency == 0:
                # Get estimation window
                est_returns = returns[i - estimation_window:i]
                cov_matrix = np.cov(est_returns, rowvar=False)
                exp_returns = np.mean(est_returns, axis=0) * 252

                # Calculate new weights based on strategy
                if strategy == "gmv":
                    # Global minimum variance
                    inv_cov = np.linalg.inv(cov_matrix)
                    ones = np.ones(n)
                    new_weights = np.dot(inv_cov, ones) / np.dot(ones, np.dot(inv_cov, ones))
                elif strategy == "equal_weight":
                    new_weights = np.array([1 / n] * n)
                elif strategy == "risk_parity":
                    result = PortfolioManagement.risk_parity_portfolio(cov_matrix)
                    new_weights = result.weights
                elif strategy == "inverse_vol":
                    vols = np.sqrt(np.diag(cov_matrix))
                    inv_vols = 1 / vols
                    new_weights = inv_vols / np.sum(inv_vols)
                elif strategy == "max_sharpe":
                    rf = kwargs.get("risk_free_rate", 0.0)
                    result = PortfolioManagement.mean_variance_optimization(
                        exp_returns, cov_matrix, rf
                    )
                    new_weights = result.weights
                else:
                    new_weights = current_weights

                current_weights = new_weights

            # Record weights
            weights_history.append(current_weights.copy())

            # Calculate portfolio return for this period
            port_ret = np.dot(current_weights, returns[i])
            portfolio_returns.append(port_ret)

        portfolio_returns = np.array(portfolio_returns)
        weights_history = np.array(weights_history)

        # Calculate turnover
        weight_changes = np.abs(np.diff(weights_history, axis=0))
        avg_turnover = np.mean(np.sum(weight_changes, axis=1)) * (252 / rebalance_frequency)

        # Performance metrics
        cumulative = np.cumprod(1 + portfolio_returns)
        total_return = cumulative[-1] - 1
        ann_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        ann_vol = np.std(portfolio_returns) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(np.min(drawdown))

        return {
            "portfolio_returns": portfolio_returns,
            "cumulative_return": cumulative.tolist(),
            "weights_history": weights_history.tolist(),
            "avg_turnover": float(avg_turnover),
            "metrics": {
                "total_return": float(total_return),
                "annualized_return": float(ann_return),
                "annualized_volatility": float(ann_vol),
                "sharpe_ratio": float(sharpe),
                "max_drawdown": max_dd,
            },
            "strategy": strategy,
            "estimation_window": estimation_window,
            "rebalance_frequency": rebalance_frequency,
        }
