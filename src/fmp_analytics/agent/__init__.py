"""Agno-based AI agent for financial analysis."""

from fmp_analytics.agent.financial_agent import FinancialAgent
from fmp_analytics.agent.tools import (
    analyze_portfolio_tool,
    analyze_stock_tool,
    get_company_profile_tool,
    get_financial_ratios_tool,
    get_historical_prices_tool,
    get_market_movers_tool,
    get_quote_tool,
    optimize_portfolio_tool,
    risk_analysis_tool,
    dcf_valuation_tool,
    options_analysis_tool,
    # Sector distortion and market explanation tools
    sector_analysis_tool,
    find_sector_distortions_tool,
    explain_market_movement_tool,
    explain_indicator_tool,
    earnings_surprise_analysis_tool,
    correlation_analysis_tool,
)

__all__ = [
    "FinancialAgent",
    "get_quote_tool",
    "get_company_profile_tool",
    "get_historical_prices_tool",
    "get_financial_ratios_tool",
    "get_market_movers_tool",
    "analyze_stock_tool",
    "analyze_portfolio_tool",
    "optimize_portfolio_tool",
    "risk_analysis_tool",
    "dcf_valuation_tool",
    "options_analysis_tool",
    # Sector distortion and market explanation tools
    "sector_analysis_tool",
    "find_sector_distortions_tool",
    "explain_market_movement_tool",
    "explain_indicator_tool",
    "earnings_surprise_analysis_tool",
    "correlation_analysis_tool",
]
