"""Analysis pipeline for financial analysis.

This module provides a structured analysis pipeline for:
- Portfolio analysis and optimization
- Risk assessment
- Valuation analysis
- Performance attribution
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import structlog

from fmp_analytics.metrics import CFAMetrics, CQFMetrics, FRMMetrics
from fmp_analytics.pipeline.data_pipeline import DataPipeline

logger = structlog.get_logger(__name__)


@dataclass
class PortfolioAnalysisResult:
    """Portfolio analysis result."""

    # Returns
    total_return: float
    annualized_return: float
    ytd_return: float
    mtd_return: float

    # Risk
    volatility: float
    var_95: float
    var_99: float
    cvar_95: float
    max_drawdown: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Relative
    beta: float
    alpha: float
    tracking_error: float
    information_ratio: float
    r_squared: float

    # Composition
    weights: dict[str, float]
    sector_allocation: dict[str, float]


@dataclass
class StockAnalysisResult:
    """Individual stock analysis result."""

    symbol: str
    company_name: str
    sector: str
    industry: str

    # Valuation
    current_price: float
    dcf_value: float
    pe_ratio: float
    pb_ratio: float
    ev_ebitda: float
    fair_value_estimate: float
    upside_potential: float

    # Fundamentals
    revenue_growth: float
    earnings_growth: float
    profit_margin: float
    roe: float
    roic: float
    debt_to_equity: float

    # Technicals
    rsi: float
    sma_50: float
    sma_200: float
    price_vs_sma_50: float
    price_vs_sma_200: float

    # Risk
    beta: float
    volatility: float
    var_95: float

    # Rating
    analyst_rating: str
    price_target: float
    overall_score: float


@dataclass
class RiskAnalysisResult:
    """Risk analysis result."""

    # VaR metrics
    var_95_1d: float
    var_99_1d: float
    var_95_10d: float
    var_99_10d: float
    cvar_95: float
    cvar_99: float

    # Volatility
    volatility_daily: float
    volatility_annual: float
    volatility_downside: float

    # Drawdown
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: int

    # Stress scenarios
    stress_test_results: dict[str, float]

    # Component risk
    component_var: dict[str, float]
    risk_contribution: dict[str, float]


class AnalysisPipeline:
    """Analysis pipeline for comprehensive financial analysis."""

    def __init__(self, data_pipeline: DataPipeline):
        """Initialize analysis pipeline.

        Args:
            data_pipeline: Data pipeline instance.
        """
        self._data = data_pipeline
        self._cfa = CFAMetrics
        self._frm = FRMMetrics
        self._cqf = CQFMetrics

    # ==================== PORTFOLIO ANALYSIS ====================

    async def analyze_portfolio(
        self,
        holdings: dict[str, float],
        benchmark_symbol: str = "SPY",
        from_date: str | date | None = None,
        to_date: str | date | None = None,
        risk_free_rate: float = 0.05,
    ) -> PortfolioAnalysisResult:
        """Perform comprehensive portfolio analysis.

        Args:
            holdings: Dictionary of symbol to weight.
            benchmark_symbol: Benchmark symbol.
            from_date: Start date for analysis.
            to_date: End date for analysis.
            risk_free_rate: Risk-free rate (annualized).

        Returns:
            PortfolioAnalysisResult dataclass.
        """
        symbols = list(holdings.keys())
        weights = np.array(list(holdings.values()))
        weights = weights / weights.sum()  # Normalize

        # Fetch returns
        returns_df = await self._data.get_returns_matrix(
            symbols + [benchmark_symbol], from_date, to_date
        )

        if returns_df.empty:
            raise ValueError("No return data available")

        # Calculate portfolio returns
        portfolio_returns = (returns_df[symbols] * weights).sum(axis=1)
        benchmark_returns = returns_df[benchmark_symbol]

        # Align data
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio_returns = aligned.iloc[:, 0].values
        benchmark_returns = aligned.iloc[:, 1].values

        # Calculate metrics
        performance = self._cfa.calculate_portfolio_performance(
            portfolio_returns, benchmark_returns, risk_free_rate
        )

        var_metrics = self._frm.calculate_var_metrics(portfolio_returns)

        # Get sector allocation (would need to fetch profiles)
        sector_allocation = {}  # Placeholder

        return PortfolioAnalysisResult(
            total_return=float(np.prod(1 + portfolio_returns) - 1),
            annualized_return=performance.returns,
            ytd_return=0.0,  # Would calculate based on year start
            mtd_return=0.0,  # Would calculate based on month start
            volatility=performance.volatility,
            var_95=var_metrics.var_95,
            var_99=var_metrics.var_99,
            cvar_95=var_metrics.cvar_95,
            max_drawdown=performance.max_drawdown,
            sharpe_ratio=performance.sharpe_ratio,
            sortino_ratio=performance.sortino_ratio,
            calmar_ratio=performance.calmar_ratio,
            beta=performance.beta,
            alpha=performance.jensens_alpha,
            tracking_error=performance.tracking_error,
            information_ratio=performance.information_ratio,
            r_squared=performance.r_squared,
            weights=dict(zip(symbols, weights.tolist())),
            sector_allocation=sector_allocation,
        )

    async def optimize_portfolio(
        self,
        symbols: list[str],
        from_date: str | date | None = None,
        to_date: str | date | None = None,
        risk_free_rate: float = 0.05,
        target_return: float | None = None,
    ) -> dict[str, Any]:
        """Optimize portfolio weights using mean-variance optimization.

        Args:
            symbols: List of stock symbols.
            from_date: Start date for historical data.
            to_date: End date for historical data.
            risk_free_rate: Risk-free rate.
            target_return: Target return (optional).

        Returns:
            Dictionary with optimal weights and metrics.
        """
        # Fetch returns
        returns_df = await self._data.get_returns_matrix(symbols, from_date, to_date)

        if returns_df.empty:
            raise ValueError("No return data available")

        # Calculate expected returns and covariance
        expected_returns = returns_df.mean().values * 252  # Annualize
        cov_matrix = returns_df.cov().values * 252  # Annualize

        # Optimize
        result = self._cqf.mean_variance_optimization(
            expected_returns, cov_matrix, risk_free_rate, target_return
        )

        return {
            "optimal_weights": dict(zip(symbols, result.weights.tolist())),
            "expected_return": result.expected_return,
            "volatility": result.volatility,
            "sharpe_ratio": result.sharpe_ratio,
        }

    async def calculate_efficient_frontier(
        self,
        symbols: list[str],
        from_date: str | date | None = None,
        to_date: str | date | None = None,
        n_points: int = 50,
    ) -> dict[str, Any]:
        """Calculate efficient frontier.

        Args:
            symbols: List of stock symbols.
            from_date: Start date.
            to_date: End date.
            n_points: Number of points on frontier.

        Returns:
            Dictionary with frontier data.
        """
        # Fetch returns
        returns_df = await self._data.get_returns_matrix(symbols, from_date, to_date)

        if returns_df.empty:
            raise ValueError("No return data available")

        expected_returns = returns_df.mean().values * 252
        cov_matrix = returns_df.cov().values * 252

        frontier_returns, frontier_vols, frontier_weights = self._cqf.efficient_frontier(
            expected_returns, cov_matrix, n_points
        )

        return {
            "returns": frontier_returns.tolist(),
            "volatilities": frontier_vols.tolist(),
            "weights": [dict(zip(symbols, w.tolist())) for w in frontier_weights],
        }

    # ==================== STOCK ANALYSIS ====================

    async def analyze_stock(
        self,
        symbol: str,
        benchmark_symbol: str = "SPY",
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> StockAnalysisResult:
        """Perform comprehensive stock analysis.

        Args:
            symbol: Stock symbol.
            benchmark_symbol: Benchmark symbol.
            from_date: Start date.
            to_date: End date.

        Returns:
            StockAnalysisResult dataclass.
        """
        # Fetch data
        profile = await self._data.get_company_profile(symbol)
        quote = await self._data.get_quote(symbol)
        ratios = await self._data.get_financial_ratios(symbol, limit=1)
        key_metrics = await self._data.get_key_metrics(symbol, limit=1)

        # Fetch historical data for risk metrics
        returns_df = await self._data.get_returns_matrix(
            [symbol, benchmark_symbol], from_date, to_date
        )

        # Calculate risk metrics
        if not returns_df.empty and symbol in returns_df.columns:
            stock_returns = returns_df[symbol].dropna().values
            benchmark_returns = returns_df[benchmark_symbol].dropna().values

            # Align
            min_len = min(len(stock_returns), len(benchmark_returns))
            stock_returns = stock_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]

            beta = self._cfa.beta(stock_returns, benchmark_returns)
            volatility = self._cfa.annualized_volatility(stock_returns)
            var_95 = self._frm.historical_var(stock_returns, 0.95)
        else:
            beta = profile.beta or 1.0
            volatility = 0.0
            var_95 = 0.0

        # Get valuation
        dcf_data = await self._data.get_dcf_valuation(symbol)
        dcf_value = dcf_data.get("dcf", quote.price or 0)

        # Extract metrics from ratios
        if not ratios.empty:
            pe_ratio = ratios.iloc[0].get("priceEarningsRatio", 0) or 0
            pb_ratio = ratios.iloc[0].get("priceToBookRatio", 0) or 0
            profit_margin = ratios.iloc[0].get("netProfitMargin", 0) or 0
            roe = ratios.iloc[0].get("returnOnEquity", 0) or 0
            debt_to_equity = ratios.iloc[0].get("debtEquityRatio", 0) or 0
        else:
            pe_ratio = quote.pe or 0
            pb_ratio = 0
            profit_margin = 0
            roe = 0
            debt_to_equity = 0

        # Extract key metrics
        if not key_metrics.empty:
            roic = key_metrics.iloc[0].get("roic", 0) or 0
            ev_ebitda = key_metrics.iloc[0].get("enterpriseValueOverEBITDA", 0) or 0
        else:
            roic = 0
            ev_ebitda = 0

        # Calculate fair value and upside
        fair_value = dcf_value
        current_price = quote.price or 0
        upside = ((fair_value / current_price) - 1) * 100 if current_price > 0 else 0

        # Calculate overall score (simplified scoring)
        score = self._calculate_stock_score(
            upside, pe_ratio, roe, profit_margin, debt_to_equity, volatility
        )

        return StockAnalysisResult(
            symbol=symbol,
            company_name=profile.company_name or symbol,
            sector=profile.sector or "Unknown",
            industry=profile.industry or "Unknown",
            current_price=current_price,
            dcf_value=dcf_value,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            ev_ebitda=ev_ebitda,
            fair_value_estimate=fair_value,
            upside_potential=upside,
            revenue_growth=0,  # Would need growth data
            earnings_growth=0,
            profit_margin=profit_margin,
            roe=roe,
            roic=roic,
            debt_to_equity=debt_to_equity,
            rsi=0,  # Would need technical indicators
            sma_50=quote.price_avg_50 or 0,
            sma_200=quote.price_avg_200 or 0,
            price_vs_sma_50=(current_price / quote.price_avg_50 - 1) * 100 if quote.price_avg_50 else 0,
            price_vs_sma_200=(current_price / quote.price_avg_200 - 1) * 100 if quote.price_avg_200 else 0,
            beta=beta,
            volatility=volatility,
            var_95=var_95,
            analyst_rating="Hold",  # Would fetch from analyst recommendations
            price_target=0,  # Would fetch from price targets
            overall_score=score,
        )

    def _calculate_stock_score(
        self,
        upside: float,
        pe_ratio: float,
        roe: float,
        profit_margin: float,
        debt_to_equity: float,
        volatility: float,
    ) -> float:
        """Calculate overall stock score.

        Args:
            upside: Upside potential percentage.
            pe_ratio: P/E ratio.
            roe: Return on equity.
            profit_margin: Net profit margin.
            debt_to_equity: Debt to equity ratio.
            volatility: Annualized volatility.

        Returns:
            Overall score (0-100).
        """
        score = 50.0  # Base score

        # Valuation (±20 points)
        if upside > 20:
            score += 20
        elif upside > 10:
            score += 10
        elif upside < -20:
            score -= 20
        elif upside < -10:
            score -= 10

        # Profitability (±15 points)
        if roe > 0.20:
            score += 15
        elif roe > 0.10:
            score += 7
        elif roe < 0:
            score -= 15

        # Margin (±10 points)
        if profit_margin > 0.20:
            score += 10
        elif profit_margin > 0.10:
            score += 5
        elif profit_margin < 0:
            score -= 10

        # Leverage (±10 points)
        if debt_to_equity < 0.5:
            score += 10
        elif debt_to_equity < 1:
            score += 5
        elif debt_to_equity > 2:
            score -= 10

        # Volatility (±5 points)
        if volatility < 0.20:
            score += 5
        elif volatility > 0.50:
            score -= 5

        return max(0, min(100, score))

    # ==================== RISK ANALYSIS ====================

    async def analyze_risk(
        self,
        holdings: dict[str, float],
        portfolio_value: float = 1000000,
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> RiskAnalysisResult:
        """Perform comprehensive risk analysis.

        Args:
            holdings: Dictionary of symbol to weight.
            portfolio_value: Total portfolio value.
            from_date: Start date.
            to_date: End date.

        Returns:
            RiskAnalysisResult dataclass.
        """
        symbols = list(holdings.keys())
        weights = np.array(list(holdings.values()))
        weights = weights / weights.sum()

        # Fetch returns
        returns_df = await self._data.get_returns_matrix(symbols, from_date, to_date)

        if returns_df.empty:
            raise ValueError("No return data available")

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1).values
        cov_matrix = returns_df.cov().values

        # VaR metrics
        var_95_1d = self._frm.historical_var(portfolio_returns, 0.95, 1)
        var_99_1d = self._frm.historical_var(portfolio_returns, 0.99, 1)
        var_95_10d = self._frm.historical_var(portfolio_returns, 0.95, 10)
        var_99_10d = self._frm.historical_var(portfolio_returns, 0.99, 10)
        cvar_95 = self._frm.conditional_var(portfolio_returns, 0.95)
        cvar_99 = self._frm.conditional_var(portfolio_returns, 0.99)

        # Volatility
        vol_daily = float(np.std(portfolio_returns, ddof=1))
        vol_annual = vol_daily * np.sqrt(252)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        vol_downside = float(np.std(downside_returns, ddof=1)) * np.sqrt(252) if len(downside_returns) > 0 else 0

        # Drawdown analysis
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(-np.min(drawdowns))
        avg_drawdown = float(-np.mean(drawdowns[drawdowns < 0])) if any(drawdowns < 0) else 0

        # Drawdown duration
        underwater = drawdowns < 0
        if any(underwater):
            # Simplified - count longest streak
            drawdown_duration = int(max(np.diff(np.where(~underwater)[0]) if np.any(~underwater) else [0]))
        else:
            drawdown_duration = 0

        # Stress test scenarios
        stress_scenarios = {
            "market_crash_10pct": np.array([-0.10] * len(symbols)),
            "market_crash_20pct": np.array([-0.20] * len(symbols)),
            "tech_selloff": np.array([-0.15 if i % 2 == 0 else -0.05 for i in range(len(symbols))]),
        }
        stress_results = self._frm.stress_test(portfolio_value, weights, stress_scenarios)

        # Component VaR
        var_decomp = self._frm.portfolio_var_decomposition(
            weights, cov_matrix, portfolio_value, 0.95
        )

        return RiskAnalysisResult(
            var_95_1d=var_95_1d * portfolio_value,
            var_99_1d=var_99_1d * portfolio_value,
            var_95_10d=var_95_10d * portfolio_value,
            var_99_10d=var_99_10d * portfolio_value,
            cvar_95=cvar_95 * portfolio_value,
            cvar_99=cvar_99 * portfolio_value,
            volatility_daily=vol_daily,
            volatility_annual=vol_annual,
            volatility_downside=vol_downside,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            drawdown_duration=drawdown_duration,
            stress_test_results=stress_results,
            component_var=dict(zip(symbols, var_decomp["component_var"].tolist())),
            risk_contribution=dict(zip(symbols, var_decomp["percent_contribution"].tolist())),
        )

    # ==================== VALUATION ANALYSIS ====================

    async def dcf_analysis(
        self,
        symbol: str,
        growth_rate: float = 0.05,
        terminal_growth: float = 0.02,
        discount_rate: float = 0.10,
        projection_years: int = 5,
    ) -> dict[str, Any]:
        """Perform DCF valuation analysis.

        Args:
            symbol: Stock symbol.
            growth_rate: Revenue/FCF growth rate.
            terminal_growth: Terminal growth rate.
            discount_rate: Discount rate (WACC).
            projection_years: Number of projection years.

        Returns:
            Dictionary with DCF analysis results.
        """
        # Fetch financial data
        profile = await self._data.get_company_profile(symbol)
        cash_flows = await self._data.get_cash_flow_statements(symbol, limit=3)
        quote = await self._data.get_quote(symbol)

        if cash_flows.empty:
            raise ValueError(f"No cash flow data for {symbol}")

        # Get latest FCF
        latest_fcf = cash_flows.iloc[0].get("freeCashFlow", 0)
        if not latest_fcf or latest_fcf <= 0:
            latest_fcf = cash_flows.iloc[0].get("operatingCashFlow", 0) - abs(
                cash_flows.iloc[0].get("capitalExpenditure", 0)
            )

        # Project FCFs
        projected_fcfs = []
        current_fcf = latest_fcf
        for year in range(1, projection_years + 1):
            current_fcf = current_fcf * (1 + growth_rate)
            projected_fcfs.append(current_fcf)

        # Calculate intrinsic value
        shares_outstanding = quote.shares_outstanding or profile.full_time_employees or 1000000

        intrinsic_value = self._cfa.dcf_valuation(
            projected_fcfs, terminal_growth, discount_rate, shares_outstanding
        )

        current_price = quote.price or 0
        upside = ((intrinsic_value / current_price) - 1) * 100 if current_price > 0 else 0

        return {
            "symbol": symbol,
            "current_price": current_price,
            "intrinsic_value": intrinsic_value,
            "upside_potential": upside,
            "assumptions": {
                "latest_fcf": latest_fcf,
                "growth_rate": growth_rate,
                "terminal_growth": terminal_growth,
                "discount_rate": discount_rate,
                "projection_years": projection_years,
            },
            "projected_fcfs": projected_fcfs,
        }

    async def comparable_analysis(
        self,
        symbol: str,
        peer_symbols: list[str] | None = None,
    ) -> dict[str, Any]:
        """Perform comparable company analysis.

        Args:
            symbol: Target stock symbol.
            peer_symbols: List of peer symbols (auto-detected if not provided).

        Returns:
            Dictionary with comparable analysis results.
        """
        # Get peers if not provided
        if not peer_symbols:
            try:
                peers_data = await self._data._company_info.get_stock_peers(symbol)
                if peers_data and peers_data[0].get("peersList"):
                    peer_symbols = peers_data[0]["peersList"][:5]
                else:
                    peer_symbols = []
            except Exception:
                peer_symbols = []

        all_symbols = [symbol] + peer_symbols

        # Fetch data for all companies
        quotes_df = await self._data.get_quotes_df(all_symbols)

        if quotes_df.empty:
            raise ValueError("No quote data available")

        # Fetch key metrics
        metrics_list = []
        for sym in all_symbols:
            try:
                metrics = await self._data.get_key_metrics(sym, limit=1)
                if not metrics.empty:
                    metrics_row = metrics.iloc[0].to_dict()
                    metrics_row["symbol"] = sym
                    metrics_list.append(metrics_row)
            except Exception:
                continue

        if not metrics_list:
            raise ValueError("No metrics data available")

        metrics_df = pd.DataFrame(metrics_list)

        # Calculate multiples comparison
        target_row = metrics_df[metrics_df["symbol"] == symbol]
        peer_rows = metrics_df[metrics_df["symbol"] != symbol]

        if target_row.empty:
            raise ValueError(f"No metrics for target {symbol}")

        # Calculate averages
        comparison = {
            "symbol": symbol,
            "current_price": quotes_df[quotes_df["symbol"] == symbol]["price"].values[0],
            "metrics": {},
            "peer_averages": {},
            "implied_values": {},
        }

        multiples = ["peRatio", "pbRatio", "priceToSalesRatio", "evToEBITDA"]

        for multiple in multiples:
            if multiple in target_row.columns:
                target_val = target_row[multiple].values[0]
                peer_avg = peer_rows[multiple].mean() if multiple in peer_rows.columns else None

                comparison["metrics"][multiple] = float(target_val) if target_val else None
                comparison["peer_averages"][multiple] = float(peer_avg) if peer_avg else None

        return comparison

    # ==================== OPTIONS ANALYSIS ====================

    async def options_analysis(
        self,
        symbol: str,
        strike: float,
        expiry_days: int,
        risk_free_rate: float = 0.05,
    ) -> dict[str, Any]:
        """Perform options analysis for a stock.

        Args:
            symbol: Stock symbol.
            strike: Strike price.
            expiry_days: Days until expiration.
            risk_free_rate: Risk-free rate.

        Returns:
            Dictionary with options analysis.
        """
        # Fetch current price and calculate implied volatility
        quote = await self._data.get_quote(symbol)
        current_price = quote.price or 0

        if current_price == 0:
            raise ValueError(f"No price data for {symbol}")

        # Fetch historical data for volatility estimation
        returns = await self._data.get_returns(symbol)
        if returns.empty:
            volatility = 0.30  # Default
        else:
            volatility = float(np.std(returns.values) * np.sqrt(252))

        # Calculate options prices
        T = expiry_days / 365
        bs_result = self._cqf.calculate_black_scholes(
            current_price, strike, T, risk_free_rate, volatility
        )

        return {
            "symbol": symbol,
            "spot_price": current_price,
            "strike_price": strike,
            "time_to_expiry": T,
            "volatility": volatility,
            "risk_free_rate": risk_free_rate,
            "call_price": bs_result.call_price,
            "put_price": bs_result.put_price,
            "greeks": {
                "delta": bs_result.greeks.delta,
                "gamma": bs_result.greeks.gamma,
                "theta": bs_result.greeks.theta,
                "vega": bs_result.greeks.vega,
                "rho": bs_result.greeks.rho,
            },
        }
