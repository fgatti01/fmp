"""Portfolio analysis and optimization tools.

Tools for portfolio analysis, optimization, and risk assessment.
"""

from fmp_analytics.agent.tools._common import (
    tool,
    get_pipeline,
    run_async,
)


@tool
def analyze_portfolio_tool(
    holdings: str,
    benchmark: str = "SPY",
) -> str:
    """Analyze a portfolio of stocks.

    Use this tool to analyze portfolio performance, risk metrics, and composition.

    Args:
        holdings: Comma-separated holdings in format "SYMBOL:WEIGHT" (e.g., "AAPL:0.3,MSFT:0.3,GOOGL:0.4")
        benchmark: Benchmark symbol (default: SPY)

    Returns:
        String with portfolio analysis results.
    """
    # Parse holdings
    holdings_dict = {}
    for item in holdings.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            symbol = parts[0].strip().upper()
            weight = float(parts[1].strip())
            holdings_dict[symbol] = weight

    if not holdings_dict:
        return "Error: Invalid holdings format. Use 'SYMBOL:WEIGHT,SYMBOL:WEIGHT'"

    data_pipeline, analysis_pipeline = get_pipeline()

    async def fetch():
        result = await analysis_pipeline.analyze_portfolio(
            holdings_dict, benchmark.upper()
        )
        await data_pipeline._client.close()
        return result

    r = run_async(fetch())

    weights_str = "\n".join([f"- {sym}: {w*100:.1f}%" for sym, w in r.weights.items()])

    return f"""
**Portfolio Analysis** (Benchmark: {benchmark})

**Allocation:**
{weights_str}

**Performance:**
- Total Return: {r.total_return*100:.2f}%
- Annualized Return: {r.annualized_return*100:.2f}%
- Volatility: {r.volatility*100:.2f}%

**Risk-Adjusted Returns:**
- Sharpe Ratio: {r.sharpe_ratio:.2f}
- Sortino Ratio: {r.sortino_ratio:.2f}
- Calmar Ratio: {r.calmar_ratio:.2f}

**Risk Metrics:**
- 95% VaR: {r.var_95*100:.2f}%
- 99% VaR: {r.var_99*100:.2f}%
- Max Drawdown: {r.max_drawdown*100:.2f}%

**Relative to Benchmark:**
- Beta: {r.beta:.2f}
- Alpha: {r.alpha*100:.2f}%
- Tracking Error: {r.tracking_error*100:.2f}%
- Information Ratio: {r.information_ratio:.2f}
- R-Squared: {r.r_squared:.2f}
"""


@tool
def optimize_portfolio_tool(
    symbols: str,
    risk_free_rate: float = 0.05,
) -> str:
    """Optimize portfolio weights using mean-variance optimization.

    Use this tool to find optimal portfolio weights that maximize the Sharpe ratio.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL,AMZN")
        risk_free_rate: Risk-free rate (default: 0.05 = 5%)

    Returns:
        String with optimal portfolio weights and expected metrics.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    data_pipeline, analysis_pipeline = get_pipeline()

    async def fetch():
        result = await analysis_pipeline.optimize_portfolio(
            symbol_list, risk_free_rate=risk_free_rate
        )
        await data_pipeline._client.close()
        return result

    r = run_async(fetch())

    weights_str = "\n".join([
        f"- {sym}: {w*100:.1f}%"
        for sym, w in sorted(r["optimal_weights"].items(), key=lambda x: -x[1])
    ])

    return f"""
**Optimal Portfolio** (Maximizing Sharpe Ratio)

**Optimal Weights:**
{weights_str}

**Expected Metrics:**
- Expected Return: {r['expected_return']*100:.2f}%
- Volatility: {r['volatility']*100:.2f}%
- Sharpe Ratio: {r['sharpe_ratio']:.2f}

*Based on historical returns. Past performance does not guarantee future results.*
"""


@tool
def risk_analysis_tool(
    holdings: str,
    portfolio_value: float = 1000000,
) -> str:
    """Perform detailed risk analysis on a portfolio.

    Use this tool to analyze VaR, stress tests, and risk contributions.

    Args:
        holdings: Comma-separated holdings in format "SYMBOL:WEIGHT"
        portfolio_value: Total portfolio value (default: $1,000,000)

    Returns:
        String with comprehensive risk analysis.
    """
    # Parse holdings
    holdings_dict = {}
    for item in holdings.split(","):
        parts = item.strip().split(":")
        if len(parts) == 2:
            symbol = parts[0].strip().upper()
            weight = float(parts[1].strip())
            holdings_dict[symbol] = weight

    if not holdings_dict:
        return "Error: Invalid holdings format. Use 'SYMBOL:WEIGHT,SYMBOL:WEIGHT'"

    data_pipeline, analysis_pipeline = get_pipeline()

    async def fetch():
        result = await analysis_pipeline.analyze_risk(
            holdings_dict, portfolio_value
        )
        await data_pipeline._client.close()
        return result

    r = run_async(fetch())

    risk_contrib_str = "\n".join([
        f"- {sym}: {contrib:.1f}%"
        for sym, contrib in sorted(r.risk_contribution.items(), key=lambda x: -x[1])
    ])

    stress_str = "\n".join([
        f"- {scenario}: ${loss:+,.0f}"
        for scenario, loss in r.stress_test_results.items()
    ])

    return f"""
**Risk Analysis** (Portfolio Value: ${portfolio_value:,.0f})

**Value at Risk (VaR):**
- 1-Day 95% VaR: ${r.var_95_1d:,.0f}
- 1-Day 99% VaR: ${r.var_99_1d:,.0f}
- 10-Day 95% VaR: ${r.var_95_10d:,.0f}
- 10-Day 99% VaR: ${r.var_99_10d:,.0f}

**Expected Shortfall (CVaR):**
- 95% CVaR: ${r.cvar_95:,.0f}
- 99% CVaR: ${r.cvar_99:,.0f}

**Volatility:**
- Daily: {r.volatility_daily*100:.2f}%
- Annual: {r.volatility_annual*100:.2f}%
- Downside: {r.volatility_downside*100:.2f}%

**Drawdown:**
- Maximum: {r.max_drawdown*100:.2f}%
- Average: {r.avg_drawdown*100:.2f}%

**Risk Contributions:**
{risk_contrib_str}

**Stress Test Results:**
{stress_str}
"""
