"""CFA (Chartered Financial Analyst) metrics calculations.

This module implements financial metrics and calculations based on the CFA curriculum:
- Portfolio Management metrics (Sharpe, Sortino, Treynor, Jensen's Alpha)
- Fixed Income analytics (Duration, Convexity, Yield measures)
- Equity Valuation (DCF, DDM, Residual Income, Multiples)
- Corporate Finance (WACC, EVA, FCF)
- Quantitative Methods (Statistics, Time Value of Money)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import newton


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

    @staticmethod
    def portfolio_return(
        weights: NDArray[np.float64],
        returns: NDArray[np.float64],
    ) -> float:
        """Calculate portfolio expected return.

        Args:
            weights: Asset weights.
            returns: Expected returns for each asset.

        Returns:
            Portfolio expected return.
        """
        return float(np.dot(weights, returns))

    @staticmethod
    def portfolio_variance(
        weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
    ) -> float:
        """Calculate portfolio variance.

        Args:
            weights: Asset weights.
            cov_matrix: Covariance matrix of returns.

        Returns:
            Portfolio variance.
        """
        return float(np.dot(weights.T, np.dot(cov_matrix, weights)))

    @staticmethod
    def portfolio_volatility(
        weights: NDArray[np.float64],
        cov_matrix: NDArray[np.float64],
    ) -> float:
        """Calculate portfolio volatility (standard deviation).

        Args:
            weights: Asset weights.
            cov_matrix: Covariance matrix of returns.

        Returns:
            Portfolio volatility.
        """
        return np.sqrt(CFAMetrics.portfolio_variance(weights, cov_matrix))

    @staticmethod
    def sharpe_ratio(
        returns: NDArray[np.float64],
        risk_free_rate: float = 0.0,
    ) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: Array of returns.
            risk_free_rate: Risk-free rate (annualized).

        Returns:
            Sharpe ratio.
        """
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))

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
        active_returns = returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)
        if tracking_error == 0:
            return 0.0
        return float(np.mean(active_returns) * 252 / tracking_error)

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
        active_returns = returns - benchmark_returns
        return float(np.std(active_returns) * np.sqrt(252))

    @staticmethod
    def beta(
        returns: NDArray[np.float64],
        market_returns: NDArray[np.float64],
    ) -> float:
        """Calculate beta coefficient.

        Args:
            returns: Array of asset returns.
            market_returns: Array of market returns.

        Returns:
            Beta coefficient.
        """
        covariance = np.cov(returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        if market_variance == 0:
            return 0.0
        return float(covariance / market_variance)

    @staticmethod
    def r_squared(
        returns: NDArray[np.float64],
        market_returns: NDArray[np.float64],
    ) -> float:
        """Calculate R-squared (coefficient of determination).

        Args:
            returns: Array of asset returns.
            market_returns: Array of market returns.

        Returns:
            R-squared value.
        """
        correlation = np.corrcoef(returns, market_returns)[0, 1]
        return float(correlation**2)

    @staticmethod
    def max_drawdown(returns: NDArray[np.float64]) -> float:
        """Calculate maximum drawdown.

        Args:
            returns: Array of returns.

        Returns:
            Maximum drawdown (as positive percentage).
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(-np.min(drawdowns))

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
        max_dd = CFAMetrics.max_drawdown(returns)
        if max_dd == 0:
            return 0.0
        annualized_return = np.mean(returns) * 252 - risk_free_rate
        return float(annualized_return / max_dd)

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

    @staticmethod
    def correlation(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """Calculate Pearson correlation coefficient.

        Args:
            x: First data series.
            y: Second data series.

        Returns:
            Correlation coefficient.
        """
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def covariance(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> float:
        """Calculate covariance.

        Args:
            x: First data series.
            y: Second data series.

        Returns:
            Covariance.
        """
        return float(np.cov(x, y)[0, 1])

    @staticmethod
    def standard_deviation(data: NDArray[np.float64]) -> float:
        """Calculate standard deviation.

        Args:
            data: Data series.

        Returns:
            Standard deviation.
        """
        return float(np.std(data, ddof=1))

    @staticmethod
    def skewness(data: NDArray[np.float64]) -> float:
        """Calculate skewness.

        Args:
            data: Data series.

        Returns:
            Skewness.
        """
        return float(stats.skew(data))

    @staticmethod
    def kurtosis(data: NDArray[np.float64]) -> float:
        """Calculate excess kurtosis.

        Args:
            data: Data series.

        Returns:
            Excess kurtosis.
        """
        return float(stats.kurtosis(data))

    @staticmethod
    def geometric_mean_return(returns: NDArray[np.float64]) -> float:
        """Calculate geometric mean return.

        Args:
            returns: Array of returns.

        Returns:
            Geometric mean return.
        """
        return float(np.prod(1 + returns) ** (1 / len(returns)) - 1)

    @staticmethod
    def annualized_return(
        returns: NDArray[np.float64],
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized return.

        Args:
            returns: Array of returns.
            periods_per_year: Number of periods per year.

        Returns:
            Annualized return.
        """
        total_return = np.prod(1 + returns) - 1
        years = len(returns) / periods_per_year
        return float((1 + total_return) ** (1 / years) - 1)

    @staticmethod
    def annualized_volatility(
        returns: NDArray[np.float64],
        periods_per_year: int = 252,
    ) -> float:
        """Calculate annualized volatility.

        Args:
            returns: Array of returns.
            periods_per_year: Number of periods per year.

        Returns:
            Annualized volatility.
        """
        return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))

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
