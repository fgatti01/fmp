"""FRM (Financial Risk Manager) metrics calculations.

This module implements risk management metrics based on the FRM curriculum:
- Market Risk: VaR, CVaR/ES, Stress Testing, GARCH
- Credit Risk: PD, LGD, EAD, Expected Loss, Merton Model
- Operational Risk: Loss Distribution Approach
- Liquidity Risk: Liquidity Coverage Ratio, Bid-Ask Spread
- Portfolio Risk: Component VaR, Marginal VaR
- Backtesting: Kupiec Test, Christoffersen Test

Based on Norte Asset Management Quant Finance Master Guide.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import minimize


@dataclass
class VaRMetrics:
    """Value at Risk metrics."""

    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    method: str


@dataclass
class CreditRiskMetrics:
    """Credit risk metrics."""

    probability_of_default: float
    loss_given_default: float
    exposure_at_default: float
    expected_loss: float
    unexpected_loss: float
    credit_var: float


@dataclass
class LiquidityMetrics:
    """Liquidity risk metrics."""

    liquidity_coverage_ratio: float
    net_stable_funding_ratio: float
    bid_ask_spread: float
    market_depth: float
    price_impact: float


class FRMMetrics:
    """FRM curriculum-based risk metrics calculator."""

    # ==================== VALUE AT RISK (VaR) ====================

    @staticmethod
    def historical_var(
        returns: NDArray[np.float64],
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """Calculate Historical Simulation VaR.

        Args:
            returns: Array of historical returns.
            confidence_level: Confidence level (e.g., 0.95 for 95%).
            holding_period: Holding period in days.

        Returns:
            VaR as a positive number.
        """
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var_1d = -sorted_returns[index]
        return var_1d * np.sqrt(holding_period)

    @staticmethod
    def parametric_var(
        returns: NDArray[np.float64],
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """Calculate Parametric (Variance-Covariance) VaR.

        Args:
            returns: Array of returns.
            confidence_level: Confidence level.
            holding_period: Holding period in days.

        Returns:
            VaR as a positive number.
        """
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z_score = stats.norm.ppf(1 - confidence_level)
        var_1d = -(mean + z_score * std)
        return var_1d * np.sqrt(holding_period)

    @staticmethod
    def monte_carlo_var(
        returns: NDArray[np.float64],
        confidence_level: float = 0.95,
        holding_period: int = 1,
        num_simulations: int = 10000,
    ) -> float:
        """Calculate Monte Carlo VaR.

        Args:
            returns: Array of historical returns.
            confidence_level: Confidence level.
            holding_period: Holding period in days.
            num_simulations: Number of Monte Carlo simulations.

        Returns:
            VaR as a positive number.
        """
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        # Generate random returns
        simulated_returns = np.random.normal(mean, std, num_simulations)

        # Scale for holding period
        simulated_returns = simulated_returns * np.sqrt(holding_period)

        # Calculate VaR
        var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)
        return float(var)

    @staticmethod
    def conditional_var(
        returns: NDArray[np.float64],
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall).

        Args:
            returns: Array of returns.
            confidence_level: Confidence level.
            holding_period: Holding period in days.

        Returns:
            CVaR as a positive number.
        """
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        cvar_1d = -np.mean(sorted_returns[:index])
        return cvar_1d * np.sqrt(holding_period)

    @staticmethod
    def parametric_cvar(
        returns: NDArray[np.float64],
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """Calculate Parametric CVaR assuming normal distribution.

        Args:
            returns: Array of returns.
            confidence_level: Confidence level.
            holding_period: Holding period in days.

        Returns:
            CVaR as a positive number.
        """
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z_score = stats.norm.ppf(1 - confidence_level)
        pdf_z = stats.norm.pdf(z_score)
        cvar_1d = -(mean - std * pdf_z / (1 - confidence_level))
        return cvar_1d * np.sqrt(holding_period)

    @staticmethod
    def component_var(
        weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        portfolio_value: float,
        confidence_level: float = 0.95,
    ) -> NDArray[np.float64]:
        """Calculate Component VaR for each asset.

        Args:
            weights: Asset weights.
            cov_matrix: Covariance matrix.
            portfolio_value: Total portfolio value.
            confidence_level: Confidence level.

        Returns:
            Array of component VaR for each asset.
        """
        portfolio_var = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        z_score = stats.norm.ppf(confidence_level)
        portfolio_var_dollar = portfolio_var * z_score * portfolio_value

        marginal_var = z_score * np.dot(cov_matrix, weights) / portfolio_var
        component_var = weights * marginal_var * portfolio_value

        return component_var

    @staticmethod
    def marginal_var(
        weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        confidence_level: float = 0.95,
    ) -> NDArray[np.float64]:
        """Calculate Marginal VaR for each asset.

        Args:
            weights: Asset weights.
            cov_matrix: Covariance matrix.
            confidence_level: Confidence level.

        Returns:
            Array of marginal VaR for each asset.
        """
        portfolio_var = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        z_score = stats.norm.ppf(confidence_level)
        marginal_var = z_score * np.dot(cov_matrix, weights) / portfolio_var
        return marginal_var

    @staticmethod
    def incremental_var(
        weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        new_weight: float,
        asset_index: int,
        confidence_level: float = 0.95,
    ) -> float:
        """Calculate Incremental VaR for adding to a position.

        Args:
            weights: Current asset weights.
            cov_matrix: Covariance matrix.
            new_weight: Additional weight to add.
            asset_index: Index of asset to add.
            confidence_level: Confidence level.

        Returns:
            Incremental VaR.
        """
        z_score = stats.norm.ppf(confidence_level)

        # Current portfolio VaR
        current_var = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        # New weights
        new_weights = weights.copy()
        new_weights[asset_index] += new_weight
        new_weights = new_weights / np.sum(new_weights)

        # New portfolio VaR
        new_var = np.sqrt(np.dot(new_weights.T, np.dot(cov_matrix, new_weights)))

        return (new_var - current_var) * z_score

    # ==================== STRESS TESTING ====================

    @staticmethod
    def stress_test(
        portfolio_value: float,
        weights: NDArray[np.float64],
        stress_scenarios: dict[str, NDArray[np.float64]],
    ) -> dict[str, float]:
        """Perform stress testing on portfolio.

        Args:
            portfolio_value: Current portfolio value.
            weights: Asset weights.
            stress_scenarios: Dictionary of scenario name to asset shocks.

        Returns:
            Dictionary of scenario name to portfolio loss.
        """
        results = {}
        for scenario_name, shocks in stress_scenarios.items():
            portfolio_return = np.dot(weights, shocks)
            loss = portfolio_value * portfolio_return
            results[scenario_name] = float(loss)
        return results

    @staticmethod
    def scenario_analysis(
        returns: NDArray[np.float64],
        historical_scenarios: dict[str, tuple[str, str]],
    ) -> dict[str, float]:
        """Analyze portfolio performance during historical scenarios.

        Args:
            returns: Full return series with dates.
            historical_scenarios: Dictionary of scenario name to (start_date, end_date).

        Returns:
            Dictionary of scenario name to cumulative return.
        """
        # This is a simplified version - in practice, dates would need to be handled
        results = {}
        for scenario_name, _ in historical_scenarios.items():
            # Placeholder - would filter returns by date range
            cumulative_return = np.prod(1 + returns) - 1
            results[scenario_name] = float(cumulative_return)
        return results

    # ==================== CREDIT RISK ====================

    @staticmethod
    def expected_loss(
        pd: float,
        lgd: float,
        ead: float,
    ) -> float:
        """Calculate Expected Loss.

        Args:
            pd: Probability of Default.
            lgd: Loss Given Default.
            ead: Exposure at Default.

        Returns:
            Expected Loss.
        """
        return pd * lgd * ead

    @staticmethod
    def unexpected_loss(
        pd: float,
        lgd: float,
        ead: float,
        lgd_volatility: float = 0.0,
    ) -> float:
        """Calculate Unexpected Loss.

        Args:
            pd: Probability of Default.
            lgd: Loss Given Default.
            ead: Exposure at Default.
            lgd_volatility: Volatility of LGD.

        Returns:
            Unexpected Loss.
        """
        variance_default = pd * (1 - pd)
        variance_lgd = lgd_volatility**2 if lgd_volatility else 0

        # UL = EAD * sqrt(pd * (1-pd) * LGD^2 + pd * var_LGD)
        ul_variance = variance_default * lgd**2 + pd * variance_lgd
        return ead * np.sqrt(ul_variance)

    @staticmethod
    def credit_var(
        pd: float,
        lgd: float,
        ead: float,
        confidence_level: float = 0.99,
        correlation: float = 0.15,
    ) -> float:
        """Calculate Credit VaR using single-factor model.

        Args:
            pd: Probability of Default.
            lgd: Loss Given Default.
            ead: Exposure at Default.
            confidence_level: Confidence level.
            correlation: Asset correlation.

        Returns:
            Credit VaR.
        """
        # Vasicek single-factor model
        z = stats.norm.ppf(confidence_level)
        pd_stressed = stats.norm.cdf(
            (stats.norm.ppf(pd) + np.sqrt(correlation) * z) / np.sqrt(1 - correlation)
        )
        el = pd * lgd * ead
        return pd_stressed * lgd * ead - el

    @staticmethod
    def merton_pd(
        asset_value: float,
        debt: float,
        asset_volatility: float,
        risk_free_rate: float,
        time_to_maturity: float,
    ) -> float:
        """Calculate PD using Merton model.

        Args:
            asset_value: Current asset value.
            debt: Face value of debt.
            asset_volatility: Asset volatility.
            risk_free_rate: Risk-free rate.
            time_to_maturity: Time to debt maturity.

        Returns:
            Probability of default.
        """
        d1 = (
            np.log(asset_value / debt)
            + (risk_free_rate + 0.5 * asset_volatility**2) * time_to_maturity
        ) / (asset_volatility * np.sqrt(time_to_maturity))

        d2 = d1 - asset_volatility * np.sqrt(time_to_maturity)

        return float(stats.norm.cdf(-d2))

    @staticmethod
    def distance_to_default(
        asset_value: float,
        debt: float,
        asset_volatility: float,
        growth_rate: float = 0.0,
    ) -> float:
        """Calculate Distance to Default.

        Args:
            asset_value: Current asset value.
            debt: Face value of debt.
            asset_volatility: Asset volatility.
            growth_rate: Expected asset growth rate.

        Returns:
            Distance to default (in standard deviations).
        """
        return (np.log(asset_value / debt) + growth_rate) / asset_volatility

    @staticmethod
    def credit_spread(
        pd: float,
        lgd: float,
        risk_free_rate: float,
    ) -> float:
        """Calculate theoretical credit spread.

        Args:
            pd: Annual probability of default.
            lgd: Loss given default.
            risk_free_rate: Risk-free rate.

        Returns:
            Credit spread.
        """
        return pd * lgd / (1 - pd * lgd)

    # ==================== OPERATIONAL RISK ====================

    @staticmethod
    def loss_distribution_approach(
        frequency_mean: float,
        frequency_std: float,
        severity_mean: float,
        severity_std: float,
        num_simulations: int = 10000,
        confidence_level: float = 0.99,
    ) -> dict[str, float]:
        """Calculate OpRisk capital using Loss Distribution Approach.

        Args:
            frequency_mean: Mean loss frequency.
            frequency_std: Std of loss frequency.
            severity_mean: Mean loss severity.
            severity_std: Std of loss severity.
            num_simulations: Number of Monte Carlo simulations.
            confidence_level: Confidence level for capital.

        Returns:
            Dictionary with expected loss, VaR, and capital.
        """
        # Simulate number of losses (Poisson)
        frequencies = np.random.poisson(frequency_mean, num_simulations)

        # Simulate total losses
        total_losses = []
        for freq in frequencies:
            if freq > 0:
                # Lognormal severity
                severities = np.random.lognormal(severity_mean, severity_std, freq)
                total_losses.append(np.sum(severities))
            else:
                total_losses.append(0)

        total_losses = np.array(total_losses)

        return {
            "expected_loss": float(np.mean(total_losses)),
            "var": float(np.percentile(total_losses, confidence_level * 100)),
            "capital": float(
                np.percentile(total_losses, confidence_level * 100) - np.mean(total_losses)
            ),
        }

    # ==================== LIQUIDITY RISK ====================

    @staticmethod
    def liquidity_coverage_ratio(
        hqla: float,
        net_cash_outflows: float,
    ) -> float:
        """Calculate Liquidity Coverage Ratio (LCR).

        Args:
            hqla: High Quality Liquid Assets.
            net_cash_outflows: Net cash outflows over 30 days.

        Returns:
            LCR ratio.
        """
        if net_cash_outflows == 0:
            return float("inf")
        return hqla / net_cash_outflows

    @staticmethod
    def net_stable_funding_ratio(
        available_stable_funding: float,
        required_stable_funding: float,
    ) -> float:
        """Calculate Net Stable Funding Ratio (NSFR).

        Args:
            available_stable_funding: Available stable funding.
            required_stable_funding: Required stable funding.

        Returns:
            NSFR ratio.
        """
        if required_stable_funding == 0:
            return float("inf")
        return available_stable_funding / required_stable_funding

    @staticmethod
    def bid_ask_spread(
        bid: float,
        ask: float,
    ) -> float:
        """Calculate bid-ask spread.

        Args:
            bid: Bid price.
            ask: Ask price.

        Returns:
            Bid-ask spread as percentage.
        """
        mid = (bid + ask) / 2
        return (ask - bid) / mid

    @staticmethod
    def amihud_illiquidity(
        returns: NDArray[np.float64],
        volumes: NDArray[np.float64],
    ) -> float:
        """Calculate Amihud illiquidity measure.

        Args:
            returns: Array of returns.
            volumes: Array of dollar volumes.

        Returns:
            Amihud illiquidity ratio.
        """
        illiquidity = np.abs(returns) / volumes
        return float(np.mean(illiquidity))

    @staticmethod
    def liquidity_adjusted_var(
        var: float,
        bid_ask_spread: float,
        position_size: float,
        daily_volume: float,
        liquidation_period: int = 10,
    ) -> float:
        """Calculate Liquidity-Adjusted VaR.

        Args:
            var: Standard VaR.
            bid_ask_spread: Bid-ask spread.
            position_size: Position size in shares.
            daily_volume: Average daily volume.
            liquidation_period: Days to liquidate.

        Returns:
            Liquidity-adjusted VaR.
        """
        # Exogenous liquidity cost (half spread)
        exogenous_cost = bid_ask_spread / 2

        # Endogenous liquidity cost (market impact)
        daily_participation = position_size / (daily_volume * liquidation_period)
        endogenous_cost = 0.1 * np.sqrt(daily_participation)  # Simplified market impact

        liquidity_adjustment = exogenous_cost + endogenous_cost
        return var + liquidity_adjustment

    # ==================== PORTFOLIO RISK ====================

    @staticmethod
    def portfolio_var_decomposition(
        weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        portfolio_value: float,
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Decompose portfolio VaR into component contributions.

        Args:
            weights: Asset weights.
            cov_matrix: Covariance matrix.
            portfolio_value: Portfolio value.
            confidence_level: Confidence level.

        Returns:
            Dictionary with VaR decomposition.
        """
        z_score = stats.norm.ppf(confidence_level)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        portfolio_var = z_score * portfolio_std * portfolio_value

        # Marginal VaR
        marginal_var = z_score * np.dot(cov_matrix, weights) / portfolio_std

        # Component VaR
        component_var = weights * marginal_var * portfolio_value

        return {
            "portfolio_var": float(portfolio_var),
            "marginal_var": marginal_var,
            "component_var": component_var,
            "percent_contribution": component_var / portfolio_var * 100,
        }

    @staticmethod
    def risk_budgeting(
        cov_matrix: NDArray[np.float64],
        risk_budgets: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Calculate weights for risk parity / risk budgeting.

        Args:
            cov_matrix: Covariance matrix.
            risk_budgets: Target risk budget for each asset.

        Returns:
            Optimal weights.
        """
        n_assets = len(risk_budgets)

        def risk_budget_objective(weights: NDArray[np.float64]) -> float:
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_std
            risk_contrib = weights * marginal_contrib
            target_contrib = risk_budgets * portfolio_std
            return float(np.sum((risk_contrib - target_contrib) ** 2))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = [(0.01, 1) for _ in range(n_assets)]
        initial_weights = np.ones(n_assets) / n_assets

        result = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        return result.x

    @staticmethod
    def calculate_var_metrics(
        returns: NDArray[np.float64],
        confidence_levels: list[float] = [0.95, 0.99],
        holding_period: int = 1,
    ) -> VaRMetrics:
        """Calculate comprehensive VaR metrics.

        Args:
            returns: Array of returns.
            confidence_levels: List of confidence levels.
            holding_period: Holding period in days.

        Returns:
            VaRMetrics dataclass.
        """
        return VaRMetrics(
            var_95=FRMMetrics.historical_var(returns, 0.95, holding_period),
            var_99=FRMMetrics.historical_var(returns, 0.99, holding_period),
            cvar_95=FRMMetrics.conditional_var(returns, 0.95, holding_period),
            cvar_99=FRMMetrics.conditional_var(returns, 0.99, holding_period),
            method="historical",
        )

    @staticmethod
    def calculate_credit_risk_metrics(
        pd: float,
        lgd: float,
        ead: float,
        lgd_volatility: float = 0.25,
        correlation: float = 0.15,
    ) -> CreditRiskMetrics:
        """Calculate comprehensive credit risk metrics.

        Args:
            pd: Probability of Default.
            lgd: Loss Given Default.
            ead: Exposure at Default.
            lgd_volatility: LGD volatility.
            correlation: Asset correlation.

        Returns:
            CreditRiskMetrics dataclass.
        """
        return CreditRiskMetrics(
            probability_of_default=pd,
            loss_given_default=lgd,
            exposure_at_default=ead,
            expected_loss=FRMMetrics.expected_loss(pd, lgd, ead),
            unexpected_loss=FRMMetrics.unexpected_loss(pd, lgd, ead, lgd_volatility),
            credit_var=FRMMetrics.credit_var(pd, lgd, ead, 0.99, correlation),
        )

    # ==================== GARCH VOLATILITY ====================

    @staticmethod
    def garch_11_volatility(
        returns: NDArray[np.float64],
        omega: float = 0.000001,
        alpha: float = 0.08,
        beta: float = 0.90,
    ) -> dict[str, Any]:
        """Calculate GARCH(1,1) conditional volatility.

        Formula:
            σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}

        Where:
            - ω: constant (long-run variance contribution)
            - α: weight of past shock (ARCH term)
            - β: weight of past variance (GARCH term)
            - α + β < 1 for stationarity

        Args:
            returns: Array of returns.
            omega: Constant term.
            alpha: ARCH coefficient.
            beta: GARCH coefficient.

        Returns:
            Dictionary with volatility series and parameters.
        """
        n = len(returns)
        variance = np.zeros(n)

        # Initial variance
        variance[0] = returns.var()

        for t in range(1, n):
            variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]

        volatility = np.sqrt(variance)

        # Long-run variance (unconditional)
        persistence = alpha + beta
        if persistence < 1:
            long_run_var = omega / (1 - persistence)
        else:
            long_run_var = variance[-1]

        return {
            "volatility": volatility,
            "variance": variance,
            "current_vol": float(volatility[-1]),
            "long_run_vol": float(np.sqrt(long_run_var)),
            "persistence": float(persistence),
            "half_life": float(np.log(0.5) / np.log(persistence)) if persistence < 1 else float("inf"),
        }

    @staticmethod
    def ewma_volatility(
        returns: NDArray[np.float64],
        lambda_decay: float = 0.94,
    ) -> NDArray[np.float64]:
        """Calculate EWMA (RiskMetrics) volatility.

        Formula:
            σ²_t = λ*σ²_{t-1} + (1-λ)*r²_{t-1}

        Args:
            returns: Array of returns.
            lambda_decay: Decay factor (typically 0.94 for daily data).

        Returns:
            Array of EWMA volatilities.
        """
        n = len(returns)
        variance = np.zeros(n)
        variance[0] = returns.var()

        for t in range(1, n):
            variance[t] = lambda_decay * variance[t - 1] + (1 - lambda_decay) * returns[t - 1] ** 2

        return np.sqrt(variance)

    # ==================== VAR BACKTESTING ====================

    @staticmethod
    def backtest_var_kupiec(
        returns: NDArray[np.float64],
        var_forecasts: NDArray[np.float64],
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Backtest VaR using Kupiec POF (Proportion of Failures) test.

        Tests whether the observed violation rate equals the expected rate.

        Formula:
            LR_POF = -2 * [ln((1-α)^(n-x) * α^x) - ln((1-p)^(n-x) * p^x)]

        Where:
            - α: expected failure rate (1 - confidence)
            - p: observed failure rate
            - n: total observations
            - x: number of violations

        Args:
            returns: Array of actual returns.
            var_forecasts: Array of VaR forecasts (positive values).
            confidence_level: VaR confidence level.

        Returns:
            Dictionary with test results.
        """
        losses = -returns
        violations = losses > var_forecasts
        n_violations = int(violations.sum())
        n_obs = len(returns)

        # Expected violations
        expected_rate = 1 - confidence_level
        expected_violations = n_obs * expected_rate

        # Observed rate
        observed_rate = n_violations / n_obs

        # Kupiec LR test statistic
        if n_violations == 0 or n_violations == n_obs:
            lr_pof = 0
        else:
            lr_pof = -2 * (
                np.log((1 - expected_rate) ** (n_obs - n_violations) * expected_rate**n_violations)
                - np.log((1 - observed_rate) ** (n_obs - n_violations) * observed_rate**n_violations)
            )

        # P-value (chi-square with 1 df)
        p_value = 1 - stats.chi2.cdf(lr_pof, df=1)

        # Basel Traffic Light
        if n_violations <= expected_violations:
            zone = "Green"
        elif n_violations <= expected_violations * 1.5:
            zone = "Yellow"
        else:
            zone = "Red"

        return {
            "violations": n_violations,
            "expected_violations": float(expected_violations),
            "violation_rate": float(observed_rate),
            "expected_rate": float(expected_rate),
            "LR_POF": float(lr_pof),
            "p_value": float(p_value),
            "reject_null": p_value < 0.05,
            "traffic_light": zone,
        }

    @staticmethod
    def backtest_var_christoffersen(
        returns: NDArray[np.float64],
        var_forecasts: NDArray[np.float64],
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Backtest VaR using Christoffersen Independence test.

        Tests whether violations are independent (not clustered).

        Args:
            returns: Array of actual returns.
            var_forecasts: Array of VaR forecasts.
            confidence_level: VaR confidence level.

        Returns:
            Dictionary with test results.
        """
        losses = -returns
        violations = (losses > var_forecasts).astype(int)

        # Transition counts
        n00 = n01 = n10 = n11 = 0

        for t in range(1, len(violations)):
            if violations[t - 1] == 0 and violations[t] == 0:
                n00 += 1
            elif violations[t - 1] == 0 and violations[t] == 1:
                n01 += 1
            elif violations[t - 1] == 1 and violations[t] == 0:
                n10 += 1
            else:
                n11 += 1

        # Transition probabilities
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        p_unconditional = (n01 + n11) / (n00 + n01 + n10 + n11)

        # LR test for independence
        if p01 == 0 or p11 == 0 or p01 == 1 or p11 == 1 or p_unconditional == 0 or p_unconditional == 1:
            lr_ind = 0
        else:
            lr_ind = -2 * (
                n00 * np.log(1 - p_unconditional)
                + n01 * np.log(p_unconditional)
                + n10 * np.log(1 - p_unconditional)
                + n11 * np.log(p_unconditional)
                - n00 * np.log(1 - p01)
                - n01 * np.log(p01)
                - n10 * np.log(1 - p11)
                - n11 * np.log(p11)
            )

        p_value = 1 - stats.chi2.cdf(lr_ind, df=1)

        return {
            "LR_independence": float(lr_ind),
            "p_value": float(p_value),
            "reject_independence": p_value < 0.05,
            "p01": float(p01),
            "p11": float(p11),
            "violations_clustered": p11 > p01,
        }

    # ==================== CREDIT PORTFOLIO RISK ====================

    @staticmethod
    def credit_portfolio_var(
        exposures: NDArray[np.float64],
        pds: NDArray[np.float64],
        lgds: NDArray[np.float64],
        correlations: NDArray[np.float64],
        confidence_level: float = 0.99,
        n_simulations: int = 10000,
    ) -> dict[str, Any]:
        """Calculate Credit VaR for a portfolio using Monte Carlo.

        Simulates correlated defaults using Gaussian copula.

        Args:
            exposures: Array of exposure amounts.
            pds: Array of probabilities of default.
            lgds: Array of loss given defaults.
            correlations: Correlation matrix for defaults.
            confidence_level: VaR confidence level.
            n_simulations: Number of Monte Carlo simulations.

        Returns:
            Dictionary with Credit VaR results.
        """
        n_obligors = len(exposures)

        # Cholesky decomposition for correlated normal variates
        L = np.linalg.cholesky(correlations)

        losses = np.zeros(n_simulations)

        for sim in range(n_simulations):
            # Generate correlated standard normal variates
            z = np.random.standard_normal(n_obligors)
            correlated_z = L @ z

            # Default if correlated variate < inverse CDF of PD
            default_thresholds = stats.norm.ppf(pds)
            defaults = correlated_z < default_thresholds

            # Calculate loss
            losses[sim] = np.sum(exposures * defaults * lgds)

        # Credit VaR
        percentile = confidence_level * 100
        credit_var = np.percentile(losses, percentile)
        expected_loss = np.mean(losses)

        return {
            "credit_var": float(credit_var),
            "expected_loss": float(expected_loss),
            "unexpected_loss": float(credit_var - expected_loss),
            "loss_std": float(np.std(losses)),
            "confidence_level": confidence_level,
        }

    # ==================== STRESS TESTING FRAMEWORK ====================

    @staticmethod
    def historical_stress_scenarios() -> dict[str, dict[str, float]]:
        """Return predefined historical stress scenarios.

        Returns:
            Dictionary of scenario names to asset class shocks.
        """
        return {
            "COVID-19 Crash (Mar 2020)": {
                "equity": -0.30,
                "fixed_income": 0.05,
                "commodities": -0.25,
                "real_estate": -0.15,
                "alternatives": -0.20,
            },
            "GFC (2008)": {
                "equity": -0.50,
                "fixed_income": 0.10,
                "commodities": -0.40,
                "real_estate": -0.35,
                "alternatives": -0.30,
            },
            "Tech Bubble (2000)": {
                "equity": -0.45,
                "fixed_income": 0.15,
                "commodities": -0.10,
                "real_estate": -0.05,
                "alternatives": -0.25,
            },
            "Interest Rate Shock (+200bp)": {
                "equity": -0.10,
                "fixed_income": -0.15,
                "commodities": 0.05,
                "real_estate": -0.12,
                "alternatives": -0.08,
            },
            "Stagflation": {
                "equity": -0.20,
                "fixed_income": -0.08,
                "commodities": 0.25,
                "real_estate": -0.10,
                "alternatives": -0.05,
            },
            "Currency Crisis": {
                "equity": -0.25,
                "fixed_income": -0.05,
                "commodities": 0.15,
                "real_estate": -0.15,
                "alternatives": -0.10,
            },
        }

    @staticmethod
    def reverse_stress_test(
        portfolio_value: float,
        weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
        loss_threshold: float,
        confidence_level: float = 0.99,
    ) -> dict[str, Any]:
        """Perform reverse stress testing.

        Find the market conditions that would cause a specified loss.

        Args:
            portfolio_value: Current portfolio value.
            weights: Asset weights.
            cov_matrix: Covariance matrix.
            loss_threshold: Loss amount to reverse engineer.
            confidence_level: Confidence level.

        Returns:
            Dictionary with reverse stress test results.
        """
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        z_score = stats.norm.ppf(confidence_level)

        # Required return shock
        required_shock = -loss_threshold / portfolio_value

        # Standard deviations required
        std_deviations = required_shock / portfolio_vol

        # Probability of this scenario
        probability = 1 - stats.norm.cdf(-std_deviations)

        return {
            "required_shock": float(required_shock),
            "standard_deviations": float(std_deviations),
            "probability": float(probability),
            "return_period_years": float(1 / probability) if probability > 0 else float("inf"),
        }
