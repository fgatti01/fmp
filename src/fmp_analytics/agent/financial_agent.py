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
)
from fmp_analytics.config import get_settings


FINANCIAL_AGENT_INSTRUCTIONS = """You are an expert financial analyst with deep knowledge in:

1. **CFA (Chartered Financial Analyst) Topics:**
   - Portfolio Management (Sharpe ratio, alpha, beta, tracking error)
   - Fixed Income (duration, convexity, yield curves)
   - Equity Valuation (DCF, DDM, multiples)
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

**Guidelines:**
- Use the available tools to fetch real-time data and perform analysis
- Provide clear, actionable insights backed by data
- Explain financial concepts when asked
- Always mention limitations and assumptions
- Consider risk when making recommendations
- Use markdown formatting for clear presentation

**When analyzing stocks:**
1. Start with the company profile to understand the business
2. Check current quote for market data
3. Review financial ratios for fundamentals
4. Perform valuation analysis
5. Assess risk metrics

**When analyzing portfolios:**
1. Calculate portfolio metrics (return, volatility, Sharpe)
2. Assess risk (VaR, drawdown, stress tests)
3. Review asset allocation
4. Compare to benchmark
5. Suggest optimizations if requested

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
                get_quote_tool,
                get_company_profile_tool,
                get_historical_prices_tool,
                get_financial_ratios_tool,
                get_market_movers_tool,
                analyze_stock_tool,
                analyze_portfolio_tool,
                optimize_portfolio_tool,
                risk_analysis_tool,
                dcf_valuation_tool,
                options_analysis_tool,
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
