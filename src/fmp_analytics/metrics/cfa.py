"""CFA (Chartered Financial Analyst) metrics calculations.

This module implements financial metrics and calculations based on the CFA curriculum:
- Portfolio Management metrics (Sharpe, Sortino, Treynor, Jensen's Alpha)
- Fixed Income analytics (Duration, Convexity, Yield measures)
- Equity Valuation (DCF, DDM, Residual Income, Multiples)
- Corporate Finance (WACC, EVA, FCF)
- Quantitative Methods (Statistics, Time Value of Money)

Note: Basic portfolio and statistical calculations delegate to metrics.core
for consistency across all modules.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import newton

from fmp_analytics.metrics import core


@dataclass
class PortfolioPerformance:
    """Portfolio performance metrics."""

    returns: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    treynor_ratio: float
    jensens_alpha: float
    information_ratio: float
    calmar_ratio: float
    max_drawdown: float
    tracking_error: float
    beta: float
    r_squared: float


@dataclass
class FixedIncomeMetrics:
    """Fixed income bond metrics."""

    ytm: float
    current_yield: float
    macaulay_duration: float
    modified_duration: float
    effective_duration: float
    convexity: float
    dv01: float
    spread_duration: float


@dataclass
class EquityValuation:
    """Equity valuation results."""

    dcf_value: float
    ddm_value: float
    residual_income_value: float
    pe_relative_value: float
    pb_relative_value: float
    ev_ebitda_value: float


class CFAMetrics:
    """CFA curriculum-based financial metrics calculator."""

    # ==================== TIME VALUE OF MONEY ====================

    @staticmethod
    def present_value(
        future_value: float,
        rate: float,
        periods: int,
    ) -> float:
        """Calculate present value.

        Args:
            future_value: Future value amount.
            rate: Discount rate per period.
            periods: Number of periods.

        Returns:
            Present value.
        """
        return future_value / ((1 + rate) ** periods)

    @staticmethod
    def future_value(
        present_value: float,
        rate: float,
        periods: int,
    ) -> float:
        """Calculate future value.

        Args:
            present_value: Present value amount.
            rate: Interest rate per period.
            periods: Number of periods.

        Returns:
            Future value.
        """
        return present_value * ((1 + rate) ** periods)

    @staticmethod
    def npv(
        cash_flows: list[float],
        discount_rate: float,
    ) -> float:
        """Calculate Net Present Value.

        Args:
            cash_flows: List of cash flows (CF0, CF1, ..., CFn).
            discount_rate: Discount rate.

        Returns:
            Net present value.
        """
        return sum(cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cash_flows))

    @staticmethod
    def irr(cash_flows: list[float], guess: float = 0.1) -> float:
        """Calculate Internal Rate of Return.

        Args:
            cash_flows: List of cash flows (CF0 is usually negative).
            guess: Initial guess for IRR.

        Returns:
            Internal rate of return.
        """

        def npv_func(rate: float) -> float:
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))

        return newton(npv_func, guess)

    @staticmethod
    def pmt(
        pv: float,
        rate: float,
        periods: int,
        fv: float = 0,
    ) -> float:
        """Calculate payment for an annuity.

        Args:
            pv: Present value.
            rate: Interest rate per period.
            periods: Number of periods.
            fv: Future value (default 0).

        Returns:
            Payment amount.
        """
        if rate == 0:
            return -(pv + fv) / periods
        pvif = (1 + rate) ** periods
        return -(pv * pvif + fv) * rate / (pvif - 1)

    # ==================== PORTFOLIO MANAGEMENT ====================

    # Delegate to core for basic portfolio calculations
    portfolio_return = staticmethod(core.portfolio_return)
    portfolio_variance = staticmethod(core.portfolio_variance)
    portfolio_volatility = staticmethod(core.portfolio_volatility)
    sharpe_ratio = staticmethod(core.sharpe_ratio)

    @staticmethod
    def sortino_ratio(
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
    ) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return).

        Args:
            returns: Array of returns.
            risk_free_rate: Risk-free rate (annualized).
            target_return: Target/minimum acceptable return.

        Returns:
            Sortino ratio.
        """
        excess_returns = returns - risk_free_rate / 252
        downside_returns = np.minimum(returns - target_return / 252, 0)
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        return float(np.mean(excess_returns) / downside_deviation * np.sqrt(252))

    @staticmethod
    def treynor_ratio(
        returns: NDArray[np.float64],
        market_returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Treynor ratio (beta-adjusted return).

        Args:
            returns: Array of portfolio returns.
            market_returns: Array of market returns.
            risk_free_rate: Risk-free rate (annualized).

        Returns:
            Treynor ratio.
        """
        beta = CFAMetrics.beta(returns, market_returns)
        if beta == 0:
            return 0.0
        excess_return = np.mean(returns) * 252 - risk_free_rate
        return float(excess_return / beta)

    @staticmethod
    def jensens_alpha(
        returns: NDArray[np.float64],
        market_returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Jensen's Alpha (CAPM-based excess return).

        Args:
            returns: Array of portfolio returns.
            market_returns: Array of market returns.
            risk_free_rate: Risk-free rate (annualized).

        Returns:
            Jensen's alpha (annualized).
        """
        beta = CFAMetrics.beta(returns, market_returns)
        portfolio_return = np.mean(returns) * 252
        market_return = np.mean(market_returns) * 252
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        return float(portfolio_return - expected_return)

    @staticmethod
    def information_ratio(
        returns: NDArray[np.float64],
        benchmark_returns: NDArray[np.float64],
    ) -> float:
        """Calculate Information ratio.

        Args:
            returns: Array of portfolio returns.
            benchmark_returns: Array of benchmark returns.

        Returns:
            Information ratio.
        """
        return core.information_ratio(returns, benchmark_returns, periods_per_year=252)

    @staticmethod
    def tracking_error(
        returns: NDArray[np.float64],
        benchmark_returns: NDArray[np.float64],
    ) -> float:
        """Calculate tracking error.

        Args:
            returns: Array of portfolio returns.
            benchmark_returns: Array of benchmark returns.

        Returns:
            Annualized tracking error.
        """
        return core.tracking_error(returns, benchmark_returns, periods_per_year=252)

    # Delegate to core for beta and r-squared calculations
    beta = staticmethod(core.beta)
    r_squared = staticmethod(core.r_squared)

    @staticmethod
    def max_drawdown(returns: NDArray[np.float64]) -> float:
        """Calculate maximum drawdown.

        Args:
            returns: Array of returns.

        Returns:
            Maximum drawdown (as positive percentage).
        """
        # Core returns negative, CFA convention returns positive
        return abs(core.max_drawdown(returns))

    @staticmethod
    def calmar_ratio(
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Calmar ratio (return over max drawdown).

        Args:
            returns: Array of returns.
            risk_free_rate: Risk-free rate (annualized).

        Returns:
            Calmar ratio.
        """
        return core.calmar_ratio(returns, risk_free_rate, periods_per_year=252)

    # ==================== FIXED INCOME ====================

    @staticmethod
    def bond_price(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """Calculate bond price.

        Args:
            face_value: Face/par value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity (annual).
            periods: Number of periods until maturity.
            frequency: Coupon payments per year (default 2 = semi-annual).

        Returns:
            Bond price.
        """
        coupon = face_value * coupon_rate / frequency
        ytm_per_period = ytm / frequency
        pv_coupons = sum(
            coupon / ((1 + ytm_per_period) ** i) for i in range(1, periods + 1)
        )
        pv_face = face_value / ((1 + ytm_per_period) ** periods)
        return pv_coupons + pv_face

    @staticmethod
    def yield_to_maturity(
        price: float,
        face_value: float,
        coupon_rate: float,
        periods: int,
        frequency: int = 2,
        guess: float = 0.05,
    ) -> float:
        """Calculate yield to maturity.

        Args:
            price: Current bond price.
            face_value: Face value of the bond.
            coupon_rate: Annual coupon rate.
            periods: Number of periods until maturity.
            frequency: Coupon payments per year.
            guess: Initial guess for YTM.

        Returns:
            Yield to maturity (annual).
        """

        def price_diff(ytm: float) -> float:
            return (
                CFAMetrics.bond_price(face_value, coupon_rate, ytm, periods, frequency)
                - price
            )

        return newton(price_diff, guess)

    @staticmethod
    def current_yield(
        annual_coupon: float,
        price: float,
    ) -> float:
        """Calculate current yield.

        Args:
            annual_coupon: Annual coupon payment.
            price: Current bond price.

        Returns:
            Current yield.
        """
        return annual_coupon / price

    @staticmethod
    def macaulay_duration(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """Calculate Macaulay duration.

        Args:
            face_value: Face value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            periods: Number of periods until maturity.
            frequency: Coupon payments per year.

        Returns:
            Macaulay duration in years.
        """
        coupon = face_value * coupon_rate / frequency
        ytm_per_period = ytm / frequency
        price = CFAMetrics.bond_price(face_value, coupon_rate, ytm, periods, frequency)

        weighted_cf = sum(
            (i * coupon) / ((1 + ytm_per_period) ** i) for i in range(1, periods + 1)
        )
        weighted_cf += (periods * face_value) / ((1 + ytm_per_period) ** periods)

        return weighted_cf / (price * frequency)

    @staticmethod
    def modified_duration(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """Calculate modified duration.

        Args:
            face_value: Face value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            periods: Number of periods until maturity.
            frequency: Coupon payments per year.

        Returns:
            Modified duration.
        """
        mac_duration = CFAMetrics.macaulay_duration(
            face_value, coupon_rate, ytm, periods, frequency
        )
        return mac_duration / (1 + ytm / frequency)

    @staticmethod
    def effective_duration(
        price: float,
        price_up: float,
        price_down: float,
        yield_change: float,
    ) -> float:
        """Calculate effective duration.

        Args:
            price: Current bond price.
            price_up: Price when yield decreases.
            price_down: Price when yield increases.
            yield_change: Yield change (e.g., 0.01 for 1%).

        Returns:
            Effective duration.
        """
        return (price_up - price_down) / (2 * price * yield_change)

    @staticmethod
    def convexity(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """Calculate convexity.

        Args:
            face_value: Face value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            periods: Number of periods until maturity.
            frequency: Coupon payments per year.

        Returns:
            Convexity.
        """
        coupon = face_value * coupon_rate / frequency
        ytm_per_period = ytm / frequency
        price = CFAMetrics.bond_price(face_value, coupon_rate, ytm, periods, frequency)

        convexity_sum = sum(
            (i * (i + 1) * coupon) / ((1 + ytm_per_period) ** (i + 2))
            for i in range(1, periods + 1)
        )
        convexity_sum += (
            periods * (periods + 1) * face_value
        ) / ((1 + ytm_per_period) ** (periods + 2))

        return convexity_sum / (price * frequency**2)

    @staticmethod
    def dv01(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        periods: int,
        frequency: int = 2,
    ) -> float:
        """Calculate DV01 (dollar value of 01 / PV01).

        Args:
            face_value: Face value of the bond.
            coupon_rate: Annual coupon rate.
            ytm: Yield to maturity.
            periods: Number of periods until maturity.
            frequency: Coupon payments per year.

        Returns:
            DV01 (price change for 1bp yield change).
        """
        price = CFAMetrics.bond_price(face_value, coupon_rate, ytm, periods, frequency)
        mod_duration = CFAMetrics.modified_duration(
            face_value, coupon_rate, ytm, periods, frequency
        )
        return price * mod_duration * 0.0001

    # ==================== EQUITY VALUATION ====================

    @staticmethod
    def dcf_valuation(
        free_cash_flows: list[float],
        terminal_growth_rate: float,
        discount_rate: float,
        shares_outstanding: float,
    ) -> float:
        """Calculate intrinsic value using DCF model.

        Args:
            free_cash_flows: Projected free cash flows.
            terminal_growth_rate: Long-term growth rate.
            discount_rate: WACC or required return.
            shares_outstanding: Number of shares outstanding.

        Returns:
            Intrinsic value per share.
        """
        # PV of explicit forecast period
        pv_fcf = sum(
            fcf / ((1 + discount_rate) ** (i + 1))
            for i, fcf in enumerate(free_cash_flows)
        )

        # Terminal value using Gordon Growth
        terminal_fcf = free_cash_flows[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
        pv_terminal = terminal_value / ((1 + discount_rate) ** len(free_cash_flows))

        enterprise_value = pv_fcf + pv_terminal
        return enterprise_value / shares_outstanding

    @staticmethod
    def gordon_growth_model(
        dividend: float,
        growth_rate: float,
        required_return: float,
    ) -> float:
        """Calculate stock value using Gordon Growth Model (DDM).

        Args:
            dividend: Next period's expected dividend (D1).
            growth_rate: Constant dividend growth rate.
            required_return: Required rate of return.

        Returns:
            Stock intrinsic value.
        """
        if required_return <= growth_rate:
            raise ValueError("Required return must exceed growth rate")
        return dividend / (required_return - growth_rate)

    @staticmethod
    def two_stage_ddm(
        d0: float,
        g1: float,
        g2: float,
        required_return: float,
        high_growth_years: int,
    ) -> float:
        """Calculate stock value using two-stage DDM.

        Args:
            d0: Current dividend.
            g1: High growth rate (stage 1).
            g2: Stable growth rate (stage 2).
            required_return: Required rate of return.
            high_growth_years: Years in high growth stage.

        Returns:
            Stock intrinsic value.
        """
        # Stage 1: High growth dividends
        pv_stage1 = sum(
            (d0 * ((1 + g1) ** t)) / ((1 + required_return) ** t)
            for t in range(1, high_growth_years + 1)
        )

        # Stage 2: Terminal value
        d_terminal = d0 * ((1 + g1) ** high_growth_years) * (1 + g2)
        terminal_value = d_terminal / (required_return - g2)
        pv_terminal = terminal_value / ((1 + required_return) ** high_growth_years)

        return pv_stage1 + pv_terminal

    @staticmethod
    def residual_income_model(
        book_value: float,
        roe: float,
        cost_of_equity: float,
        growth_rate: float,
    ) -> float:
        """Calculate stock value using Residual Income Model.

        Args:
            book_value: Current book value per share.
            roe: Return on equity.
            cost_of_equity: Cost of equity capital.
            growth_rate: Growth rate of residual income.

        Returns:
            Stock intrinsic value.
        """
        residual_income = book_value * (roe - cost_of_equity)
        pv_residual_income = residual_income / (cost_of_equity - growth_rate)
        return book_value + pv_residual_income

    # ==================== CORPORATE FINANCE ====================

    @staticmethod
    def wacc(
        equity_weight: float,
        debt_weight: float,
        cost_of_equity: float,
        cost_of_debt: float,
        tax_rate: float,
    ) -> float:
        """Calculate Weighted Average Cost of Capital.

        Args:
            equity_weight: Weight of equity in capital structure.
            debt_weight: Weight of debt in capital structure.
            cost_of_equity: Cost of equity.
            cost_of_debt: Pre-tax cost of debt.
            tax_rate: Corporate tax rate.

        Returns:
            WACC.
        """
        return (equity_weight * cost_of_equity) + (
            debt_weight * cost_of_debt * (1 - tax_rate)
        )

    @staticmethod
    def capm(
        risk_free_rate: float,
        beta: float,
        market_return: float,
    ) -> float:
        """Calculate expected return using CAPM.

        Args:
            risk_free_rate: Risk-free rate.
            beta: Asset beta.
            market_return: Expected market return.

        Returns:
            Expected return.
        """
        return risk_free_rate + beta * (market_return - risk_free_rate)

    @staticmethod
    def economic_value_added(
        nopat: float,
        invested_capital: float,
        wacc: float,
    ) -> float:
        """Calculate Economic Value Added (EVA).

        Args:
            nopat: Net Operating Profit After Tax.
            invested_capital: Total invested capital.
            wacc: Weighted average cost of capital.

        Returns:
            EVA.
        """
        return nopat - (invested_capital * wacc)

    @staticmethod
    def free_cash_flow_to_firm(
        ebit: float,
        tax_rate: float,
        depreciation: float,
        capex: float,
        change_in_nwc: float,
    ) -> float:
        """Calculate Free Cash Flow to Firm (FCFF).

        Args:
            ebit: Earnings before interest and taxes.
            tax_rate: Tax rate.
            depreciation: Depreciation and amortization.
            capex: Capital expenditures.
            change_in_nwc: Change in net working capital.

        Returns:
            FCFF.
        """
        return ebit * (1 - tax_rate) + depreciation - capex - change_in_nwc

    @staticmethod
    def free_cash_flow_to_equity(
        net_income: float,
        depreciation: float,
        capex: float,
        change_in_nwc: float,
        net_borrowing: float,
    ) -> float:
        """Calculate Free Cash Flow to Equity (FCFE).

        Args:
            net_income: Net income.
            depreciation: Depreciation and amortization.
            capex: Capital expenditures.
            change_in_nwc: Change in net working capital.
            net_borrowing: Net borrowing (new debt - debt repaid).

        Returns:
            FCFE.
        """
        return net_income + depreciation - capex - change_in_nwc + net_borrowing

    # ==================== QUANTITATIVE METHODS ====================

    # Delegate to core for basic statistics
    correlation = staticmethod(core.correlation)
    covariance = staticmethod(core.covariance)
    standard_deviation = staticmethod(core.std)
    skewness = staticmethod(core.skewness)
    kurtosis = staticmethod(core.kurtosis)

    # ==================== EDHEC RISK METRICS ====================

    # Delegate to core for basic risk calculations
    semideviation = staticmethod(core.semideviation)
    downside_deviation = staticmethod(core.downside_deviation)

    @staticmethod
    def cornish_fisher_var(
        returns: NDArray[np.float64],
        confidence_level: float = 0.95,
    ) -> dict[str, Any]:
        """Calculate Cornish-Fisher adjusted VaR.

        The Cornish-Fisher expansion adjusts the normal distribution z-score
        for skewness and kurtosis, providing more accurate VaR for non-normal
        distributions.

        Formula:
            z_cf = z + (z² - 1)S/6 + (z³ - 3z)(K - 3)/24 - (2z³ - 5z)S²/36

        Where:
            z = standard normal z-score
            S = skewness
            K = kurtosis

        Args:
            returns: Array of returns.
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR).

        Returns:
            Dictionary with:
                - gaussian_var: Standard parametric VaR
                - cornish_fisher_var: Adjusted VaR for non-normality
                - z_score: Normal z-score
                - z_cf: Adjusted z-score
                - skewness: Sample skewness
                - kurtosis: Sample excess kurtosis
                - adjustment: Difference between CF and Gaussian VaR
        """
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)  # Excess kurtosis

        # Standard normal z-score
        z = stats.norm.ppf(1 - confidence_level)

        # Cornish-Fisher expansion
        z_cf = (
            z
            + (z**2 - 1) * skew / 6
            + (z**3 - 3 * z) * kurt / 24
            - (2 * z**3 - 5 * z) * skew**2 / 36
        )

        # Calculate VaR values
        gaussian_var = -(mean + z * std)
        cornish_fisher_var = -(mean + z_cf * std)

        return {
            "gaussian_var": float(gaussian_var),
            "cornish_fisher_var": float(cornish_fisher_var),
            "z_score": float(z),
            "z_cf": float(z_cf),
            "skewness": float(skew),
            "kurtosis": float(kurt),
            "adjustment": float(cornish_fisher_var - gaussian_var),
            "confidence_level": confidence_level,
        }

    @staticmethod
    def jarque_bera_test(
        returns: NDArray[np.float64],
        significance_level: float = 0.01,
    ) -> dict[str, Any]:
        """Perform Jarque-Bera test for normality.

        The JB test uses skewness and kurtosis to test whether returns
        come from a normal distribution.

        Formula: JB = (n/6) × [S² + (K²/4)]

        Where:
            n = sample size
            S = skewness
            K = excess kurtosis

        Args:
            returns: Array of returns.
            significance_level: Significance level for the test.

        Returns:
            Dictionary with:
                - statistic: JB test statistic
                - p_value: p-value of the test
                - is_normal: True if we cannot reject normality
                - skewness: Sample skewness
                - kurtosis: Sample excess kurtosis
        """
        statistic, p_value = stats.jarque_bera(returns)
        skew = stats.skew(returns)
        kurt = stats.kurtosis(returns)

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > significance_level,
            "skewness": float(skew),
            "kurtosis": float(kurt),
            "significance_level": significance_level,
            "interpretation": (
                "Cannot reject normality" if p_value > significance_level
                else "Reject normality (returns are non-normal)"
            ),
        }

    @staticmethod
    def comprehensive_drawdown(
        returns: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Calculate comprehensive drawdown statistics.

        Provides wealth index, drawdown series, and key drawdown metrics.

        Args:
            returns: Array of returns.

        Returns:
            Dictionary with:
                - wealth_index: Cumulative wealth (starting at 1)
                - previous_peak: Running maximum wealth
                - drawdown: Percentage drawdown from peak
                - max_drawdown: Maximum drawdown
                - max_drawdown_start: Index where max drawdown started
                - max_drawdown_end: Index where max drawdown ended
                - avg_drawdown: Average drawdown
                - drawdown_duration: Length of max drawdown period
                - current_drawdown: Current drawdown from peak
                - recovery_time: Periods to recover from max drawdown (None if not recovered)
        """
        # Calculate wealth index
        wealth_index = np.cumprod(1 + returns)
        previous_peak = np.maximum.accumulate(wealth_index)
        drawdown = (wealth_index - previous_peak) / previous_peak

        # Find max drawdown
        max_dd = float(np.min(drawdown))
        max_dd_end_idx = int(np.argmin(drawdown))

        # Find start of max drawdown (last peak before max dd)
        wealth_at_max_dd_end = wealth_index[max_dd_end_idx]
        peak_before_max_dd = previous_peak[max_dd_end_idx]

        # Find where this peak occurred
        max_dd_start_idx = int(np.where(wealth_index[:max_dd_end_idx + 1] == peak_before_max_dd)[0][-1])

        # Find recovery time (when wealth returns to previous peak)
        recovery_time = None
        if max_dd_end_idx < len(wealth_index) - 1:
            post_trough = wealth_index[max_dd_end_idx:]
            recovered_idx = np.where(post_trough >= peak_before_max_dd)[0]
            if len(recovered_idx) > 0:
                recovery_time = int(recovered_idx[0])

        return {
            "wealth_index": wealth_index.tolist(),
            "previous_peak": previous_peak.tolist(),
            "drawdown": drawdown.tolist(),
            "max_drawdown": max_dd,
            "max_drawdown_start": max_dd_start_idx,
            "max_drawdown_end": max_dd_end_idx,
            "avg_drawdown": float(np.mean(drawdown[drawdown < 0])) if np.any(drawdown < 0) else 0.0,
            "drawdown_duration": max_dd_end_idx - max_dd_start_idx,
            "current_drawdown": float(drawdown[-1]),
            "recovery_time": recovery_time,
        }

    @staticmethod
    def summary_stats(
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> dict[str, Any]:
        """Calculate comprehensive summary statistics (EDHEC-style).

        Provides all key risk and return metrics in a single function.

        Args:
            returns: Array of returns.
            risk_free_rate: Annual risk-free rate.
            periods_per_year: Trading periods per year (252 for daily).

        Returns:
            Dictionary with comprehensive statistics.
        """
        # Annualize returns and volatility
        ann_return = CFAMetrics.annualized_return(returns, periods_per_year)
        ann_vol = CFAMetrics.annualized_volatility(returns, periods_per_year)

        # Risk metrics
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns))

        # Period risk-free rate
        rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = returns - rf_per_period

        # Sharpe ratio
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

        # Cornish-Fisher VaR
        cf_var = CFAMetrics.cornish_fisher_var(returns, 0.95)

        # CVaR
        var_threshold = np.percentile(returns, 5)
        cvar = float(np.mean(returns[returns <= var_threshold]))

        # Max Drawdown
        max_dd = CFAMetrics.max_drawdown(returns)

        return {
            "annualized_return": ann_return,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "skewness": skew,
            "kurtosis": kurt,
            "historic_var_5": float(np.percentile(returns, 5)),
            "cornish_fisher_var_5": cf_var["cornish_fisher_var"],
            "historic_cvar_5": cvar,
            "max_drawdown": max_dd,
            "semideviation": CFAMetrics.semideviation(returns),
            "is_normal": CFAMetrics.jarque_bera_test(returns)["is_normal"],
        }

    # Delegate to core for return calculations
    geometric_mean_return = staticmethod(core.geometric_mean_return)
    annualized_return = staticmethod(core.annualize_return)
    annualized_volatility = staticmethod(core.annualize_volatility)

    @staticmethod
    def calculate_portfolio_performance(
        returns: NDArray[np.float64],
        market_returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
    ) -> PortfolioPerformance:
        """Calculate comprehensive portfolio performance metrics.

        Args:
            returns: Array of portfolio returns.
            market_returns: Array of market returns.
            risk_free_rate: Risk-free rate (annualized).

        Returns:
            PortfolioPerformance dataclass with all metrics.
        """
        return PortfolioPerformance(
            returns=CFAMetrics.annualized_return(returns),
            volatility=CFAMetrics.annualized_volatility(returns),
            sharpe_ratio=CFAMetrics.sharpe_ratio(returns, risk_free_rate),
            sortino_ratio=CFAMetrics.sortino_ratio(returns, risk_free_rate),
            treynor_ratio=CFAMetrics.treynor_ratio(returns, market_returns, risk_free_rate),
            jensens_alpha=CFAMetrics.jensens_alpha(returns, market_returns, risk_free_rate),
            information_ratio=CFAMetrics.information_ratio(returns, market_returns),
            calmar_ratio=CFAMetrics.calmar_ratio(returns, risk_free_rate),
            max_drawdown=CFAMetrics.max_drawdown(returns),
            tracking_error=CFAMetrics.tracking_error(returns, market_returns),
            beta=CFAMetrics.beta(returns, market_returns),
            r_squared=CFAMetrics.r_squared(returns, market_returns),
        )
