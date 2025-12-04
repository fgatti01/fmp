"""Financial analysis tools for Agno agent.

This package organizes tools into logical categories:

- data_tools: Quote, profile, historical prices, financial ratios, market movers
- analysis_tools: Stock analysis
- portfolio_tools: Portfolio analysis, optimization, risk analysis
- valuation_tools: DCF valuation, options pricing
- market_tools: Sector analysis, distortion detection, market movement explanation
- comprehensive_tools: In-depth equity/sector analysis, earnings, correlation
- advanced_tools: Norte/EDHEC portfolio methods (Risk Parity, Black-Litterman, etc.)
"""

# Data retrieval tools
from fmp_analytics.agent.tools.data_tools import (
    get_quote_tool,
    get_company_profile_tool,
    get_historical_prices_tool,
    get_financial_ratios_tool,
    get_market_movers_tool,
)

# Stock analysis tools
from fmp_analytics.agent.tools.analysis_tools import (
    analyze_stock_tool,
)

# Portfolio tools
from fmp_analytics.agent.tools.portfolio_tools import (
    analyze_portfolio_tool,
    optimize_portfolio_tool,
    risk_analysis_tool,
)

# Valuation tools
from fmp_analytics.agent.tools.valuation_tools import (
    dcf_valuation_tool,
    options_analysis_tool,
)

# Market and sector tools
from fmp_analytics.agent.tools.market_tools import (
    sector_analysis_tool,
    find_sector_distortions_tool,
    explain_market_movement_tool,
    explain_indicator_tool,
)

# Comprehensive analysis tools
from fmp_analytics.agent.tools.comprehensive_tools import (
    earnings_surprise_analysis_tool,
    comprehensive_equity_analysis_tool,
    comprehensive_sector_analysis_tool,
    correlation_analysis_tool,
)

# Advanced portfolio management tools (Norte/EDHEC)
from fmp_analytics.agent.tools.advanced_tools import (
    risk_parity_portfolio_tool,
    black_litterman_portfolio_tool,
    kelly_criterion_tool,
    monte_carlo_wealth_tool,
    advanced_portfolio_optimization_tool,
    glide_path_tool,
)


__all__ = [
    # Data tools
    "get_quote_tool",
    "get_company_profile_tool",
    "get_historical_prices_tool",
    "get_financial_ratios_tool",
    "get_market_movers_tool",
    # Analysis tools
    "analyze_stock_tool",
    # Portfolio tools
    "analyze_portfolio_tool",
    "optimize_portfolio_tool",
    "risk_analysis_tool",
    # Valuation tools
    "dcf_valuation_tool",
    "options_analysis_tool",
    # Market tools
    "sector_analysis_tool",
    "find_sector_distortions_tool",
    "explain_market_movement_tool",
    "explain_indicator_tool",
    # Comprehensive tools
    "earnings_surprise_analysis_tool",
    "comprehensive_equity_analysis_tool",
    "comprehensive_sector_analysis_tool",
    "correlation_analysis_tool",
    # Advanced portfolio tools
    "risk_parity_portfolio_tool",
    "black_litterman_portfolio_tool",
    "kelly_criterion_tool",
    "monte_carlo_wealth_tool",
    "advanced_portfolio_optimization_tool",
    "glide_path_tool",
]
