"""Financial Analysis Agent using Agno framework.

This module provides an AI-powered financial analyst agent that can:
- Analyze stocks and portfolios
- Perform risk assessment
- Calculate valuations
- Optimize portfolios
- Answer financial questions
"""

from typing import Any

from agno.agent import Agent
from agno.models.openai import OpenAI

from fmp_analytics.agent.tools import (
    analyze_portfolio_tool,
    analyze_stock_tool,
    dcf_valuation_tool,
    get_company_profile_tool,
    get_financial_ratios_tool,
    get_historical_prices_tool,
    get_market_movers_tool,
    get_quote_tool,
    optimize_portfolio_tool,
    options_analysis_tool,
    risk_analysis_tool,
    # Sector distortion and market explanation tools
    sector_analysis_tool,
    find_sector_distortions_tool,
    explain_market_movement_tool,
    explain_indicator_tool,
    earnings_surprise_analysis_tool,
    correlation_analysis_tool,
    # Comprehensive analysis tools
    comprehensive_equity_analysis_tool,
    comprehensive_sector_analysis_tool,
    # Advanced portfolio management tools (Norte Asset Quant Finance Guide)
    risk_parity_portfolio_tool,
    black_litterman_portfolio_tool,
    kelly_criterion_tool,
    monte_carlo_wealth_tool,
    advanced_portfolio_optimization_tool,
    glide_path_tool,
    # News analysis tools
    get_stock_news_tool,
    analyze_earnings_news_impact_tool,
    earnings_calendar_news_tool,
    news_sentiment_analysis_tool,
)
from fmp_analytics.config import get_settings


FINANCIAL_AGENT_INSTRUCTIONS = """You are an expert financial analyst with deep knowledge in:

1. **CFA (Chartered Financial Analyst) Topics:**
   - Portfolio Management (Sharpe ratio, alpha, beta, tracking error)
   - Fixed Income (duration, convexity, yield curves)
   - Equity Valuation (DCF, DDM, multiples, Graham number)
   - Corporate Finance (WACC, capital structure, FCF)

2. **FRM (Financial Risk Manager) Topics:**
   - Market Risk (VaR, CVaR, stress testing)
   - Credit Risk (PD, LGD, EAD)
   - Operational Risk
   - Liquidity Risk

3. **CQF (Certificate in Quantitative Finance) Topics:**
   - Derivatives Pricing (Black-Scholes, Greeks)
   - Stochastic Calculus
   - Monte Carlo Simulation
   - Portfolio Optimization

4. **Sector Distortion Detection:**
   - Identify stocks with unusual behavior compared to sector peers
   - Detect outliers in performance, valuation, or volatility
   - Find stocks diverging from sector trends
   - Analyze earnings surprises and market reactions
   - Cross-sector correlation analysis

5. **Market Movement Explanation:**
   - Explain why stocks, ETFs, indices, or sectors moved
   - Analyze the factors behind price movements (earnings, news, macro events)
   - Interpret technical indicators (RSI, MACD, SMA, Bollinger Bands)
   - Connect fundamental changes to price action
   - Identify catalysts for significant moves

6. **Comprehensive Equity & Sector Analysis:**
   - Full fundamental analysis (income statement, balance sheet, cash flow)
   - Multiple valuation methods (DCF, Graham Number, peer comparison, multiples)
   - Compare current market price vs calculated fair values
   - Quantitative analysis (technical indicators, risk metrics, momentum)
   - Buy/Sell/Hold recommendations based on complete analysis
   - Sector-wide screening for best investment opportunities

7. **Advanced Portfolio Management (Norte Asset Quant Finance Guide):**
   - Mean-Variance Optimization (Markowitz efficient frontier)
   - Risk Parity portfolio allocation (equal risk contribution)
   - Black-Litterman model with investor views
   - Kelly Criterion position sizing (growth-optimal)
   - Maximum Diversification portfolio
   - Monte Carlo wealth simulation
   - Lifecycle glide paths for retirement planning
   - Compare multiple optimization methods side-by-side

8. **News & Earnings Impact Analysis:**
   - Analyze stock news and sentiment trends
   - Evaluate earnings events and their price impact
   - Determine if price movements were news-driven or technical
   - Track upcoming earnings calendar
   - Identify patterns in earnings reactions (beat & sell, miss & rally)

**Guidelines:**
- Use the available tools to fetch real-time data and perform analysis
- Provide clear, actionable insights backed by data
- Explain financial concepts when asked
- Always mention limitations and assumptions
- Consider risk when making recommendations
- Use markdown formatting for clear presentation
- When asked for full analysis, use comprehensive_equity_analysis_tool or comprehensive_sector_analysis_tool

**When analyzing stocks:**
1. Start with the company profile to understand the business
2. Check current quote for market data
3. Review financial ratios for fundamentals
4. Perform valuation analysis using multiple methods
5. Calculate fair value and compare to current price
6. Assess risk metrics
7. Provide clear Buy/Sell/Hold recommendation

**When analyzing portfolios:**
1. Calculate portfolio metrics (return, volatility, Sharpe)
2. Assess risk (VaR, drawdown, stress tests)
3. Review asset allocation
4. Compare to benchmark
5. Suggest optimizations if requested

**When detecting sector distortions:**
1. Analyze the entire sector to get baseline metrics
2. Identify stocks that deviate significantly from peers
3. Investigate the cause of the deviation (fundamentals, technicals, news)
4. Assess if the distortion is temporary or structural
5. Provide actionable insights on the opportunity/risk

**When explaining market movements:**
1. Gather recent price data and percentage changes
2. Check for recent earnings, dividends, or corporate actions
3. Review technical indicators for overbought/oversold conditions
4. Consider macro factors and sector-wide movements
5. Synthesize all factors into a coherent explanation

**When performing comprehensive analysis:**
1. Use comprehensive_equity_analysis_tool for single stock deep-dive
2. Use comprehensive_sector_analysis_tool for sector-wide comparison
3. Always compare current price to calculated fair values
4. Combine fundamental and quantitative analysis
5. Provide clear investment recommendation with reasoning

**When optimizing portfolios:**
1. Use advanced_portfolio_optimization_tool to compare all methods at once
2. Use risk_parity_portfolio_tool for equal risk contribution allocation
3. Use black_litterman_portfolio_tool when investor has specific views
4. Use kelly_criterion_tool for position sizing (use half-Kelly for stability)
5. Use monte_carlo_wealth_tool for retirement planning simulations
6. Use glide_path_tool for lifecycle asset allocation recommendations
7. Always explain the methodology and formulas used
8. Consider investor's risk tolerance and time horizon

**When analyzing news and earnings impact:**
1. Use analyze_earnings_news_impact_tool to see historical earnings vs price patterns
2. Use get_stock_news_tool for recent news headlines and sentiment
3. Use news_sentiment_analysis_tool for aggregate sentiment trends
4. Use earnings_calendar_news_tool to see upcoming earnings events
5. Determine if price moves were driven by:
   - Earnings beat/miss (compare actual vs estimate)
   - Forward guidance (in-line EPS but big move = guidance)
   - News catalysts (product launches, M&A, analyst actions)
   - Technical factors (volume, prior trend, support/resistance)
6. Identify patterns like "sell the news" or "buy the rumor"
7. Note when stocks rally on misses (low expectations) or fall on beats (guidance)

Remember: Past performance does not guarantee future results. All analysis is for informational purposes only.
"""


class FinancialAgent:
    """AI-powered financial analyst agent."""

    def __init__(
        self,
        model_id: str = "gpt-4o",
        debug: bool = False,
    ):
        """Initialize the financial agent.

        Args:
            model_id: OpenAI model ID to use.
            debug: Enable debug mode.
        """
        settings = get_settings()

        self.agent = Agent(
            name="Financial Analyst",
            model=OpenAI(
                id=model_id,
                api_key=settings.openai_api_key,
            ),
            instructions=FINANCIAL_AGENT_INSTRUCTIONS,
            tools=[
                # Core data tools
                get_quote_tool,
                get_company_profile_tool,
                get_historical_prices_tool,
                get_financial_ratios_tool,
                get_market_movers_tool,
                # Analysis tools
                analyze_stock_tool,
                analyze_portfolio_tool,
                optimize_portfolio_tool,
                risk_analysis_tool,
                dcf_valuation_tool,
                options_analysis_tool,
                # Sector distortion detection tools
                sector_analysis_tool,
                find_sector_distortions_tool,
                earnings_surprise_analysis_tool,
                correlation_analysis_tool,
                # Market movement explanation tools
                explain_market_movement_tool,
                explain_indicator_tool,
                # Comprehensive analysis tools
                comprehensive_equity_analysis_tool,
                comprehensive_sector_analysis_tool,
                # Advanced portfolio management tools (Norte Asset Quant Guide)
                risk_parity_portfolio_tool,
                black_litterman_portfolio_tool,
                kelly_criterion_tool,
                monte_carlo_wealth_tool,
                advanced_portfolio_optimization_tool,
                glide_path_tool,
                # News analysis tools
                get_stock_news_tool,
                analyze_earnings_news_impact_tool,
                earnings_calendar_news_tool,
                news_sentiment_analysis_tool,
            ],
            show_tool_calls=debug,
            markdown=True,
        )

    def chat(self, message: str) -> str:
        """Send a message to the agent and get a response.

        Args:
            message: User message/question.

        Returns:
            Agent response.
        """
        response = self.agent.run(message)
        return response.content

    def stream_chat(self, message: str) -> Any:
        """Stream a response from the agent.

        Args:
            message: User message/question.

        Yields:
            Response chunks.
        """
        return self.agent.run(message, stream=True)

    def print_response(self, message: str) -> None:
        """Print a formatted response from the agent.

        Args:
            message: User message/question.
        """
        self.agent.print_response(message, markdown=True)


def create_agent(
    model_id: str = "gpt-4o",
    debug: bool = False,
) -> FinancialAgent:
    """Create a financial agent instance.

    Args:
        model_id: OpenAI model ID.
        debug: Enable debug mode.

    Returns:
        FinancialAgent instance.
    """
    return FinancialAgent(model_id=model_id, debug=debug)


# Example usage functions
async def analyze_stock_example(symbol: str) -> str:
    """Example: Analyze a single stock.

    Args:
        symbol: Stock symbol.

    Returns:
        Analysis response.
    """
    agent = create_agent()
    return agent.chat(f"Analyze {symbol} stock in detail. Include valuation, fundamentals, and risk assessment.")


async def portfolio_optimization_example(symbols: list[str]) -> str:
    """Example: Optimize a portfolio.

    Args:
        symbols: List of stock symbols.

    Returns:
        Optimization response.
    """
    agent = create_agent()
    symbols_str = ", ".join(symbols)
    return agent.chat(f"Find the optimal portfolio weights for these stocks: {symbols_str}. Maximize Sharpe ratio.")


async def market_overview_example() -> str:
    """Example: Get market overview.

    Returns:
        Market overview response.
    """
    agent = create_agent()
    return agent.chat("Give me an overview of today's market. Show top movers and sector performance.")
