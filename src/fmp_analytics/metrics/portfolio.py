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
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize


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
