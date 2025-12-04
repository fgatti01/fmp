"""Advanced portfolio management tools.

Tools based on Norte Asset Management Quant Finance Guide and EDHEC methods
for advanced portfolio optimization including Risk Parity, Black-Litterman,
Kelly Criterion, Monte Carlo simulation, and lifecycle planning.
"""

from fmp_analytics.agent.tools._common import (
    np,
    pd,
    tool,
    get_pipeline,
    run_async,
    PortfolioManagement,
)

@tool
def risk_parity_portfolio_tool(
    symbols: str,
    risk_budgets: str = "",
) -> str:
    """Optimize portfolio using Risk Parity methodology.

    Risk Parity allocates weights so each asset contributes equally to portfolio risk.
    This approach provides better diversification than equal-weight or mean-variance.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL,BND,GLD")
        risk_budgets: Optional comma-separated risk budgets (e.g., "0.25,0.25,0.25,0.25")
                     If empty, uses equal risk contribution.

    Returns:
        String with risk parity portfolio weights and analysis.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    # Parse risk budgets if provided
    budgets = None
    if risk_budgets.strip():
        try:
            budgets = np.array([float(b.strip()) for b in risk_budgets.split(",")])
            budgets = budgets / budgets.sum()  # Normalize
        except ValueError:
            budgets = None

    data_pipeline, _ = get_pipeline()

    async def fetch():
        prices = await data_pipeline.get_historical_prices_batch(symbol_list, limit=252)
        await data_pipeline._client.close()
        return prices

    prices_dict = run_async(fetch())

    # Build returns dataframe
    import pandas as pd
    returns_data = {}
    for symbol, prices in prices_dict.items():
        if prices:
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            returns = np.diff(np.log(closes))
            returns_data[symbol] = returns

    if len(returns_data) < 2:
        return "Error: Need at least 2 symbols with valid data for optimization."

    # Align all returns to same length
    min_len = min(len(r) for r in returns_data.values())
    for sym in returns_data:
        returns_data[sym] = returns_data[sym][-min_len:]

    returns_df = pd.DataFrame(returns_data)
    cov_matrix = returns_df.cov().values * 252  # Annualize

    # Calculate risk parity portfolio
    result_data = PortfolioManagement.risk_parity_portfolio(
        cov_matrix=cov_matrix,
        risk_budgets=budgets,
    )

    # Build result string
    result = "**Risk Parity Portfolio Optimization**\n\n"
    result += "**Methodology:**\n"
    result += "Risk Parity allocates portfolio weights so that each asset contributes\n"
    result += "equally (or according to specified budgets) to total portfolio risk.\n\n"

    result += "**Formula:**\n"
    result += "```\n"
    result += "Minimize: Σᵢ(RCᵢ - bᵢ)²\n"
    result += "Where: RCᵢ = wᵢ × (Σw)ᵢ / σₚ (Risk Contribution of asset i)\n"
    result += "       bᵢ = target risk budget for asset i\n"
    result += "       σₚ = portfolio volatility\n"
    result += "```\n\n"

    result += "**Optimal Weights:**\n"
    for i, sym in enumerate(returns_df.columns):
        weight = result_data.weights[i]
        rc = result_data.risk_contributions[i]
        result += f"- {sym}: {weight*100:.1f}% (Risk Contribution: {rc*100:.1f}%)\n"

    result += f"\n**Portfolio Metrics:**\n"
    result += f"- Portfolio Volatility: {result_data.portfolio_volatility*100:.2f}%\n"

    # Show individual asset volatilities for comparison
    result += "\n**Individual Asset Volatilities:**\n"
    asset_vols = np.sqrt(np.diag(cov_matrix))
    for i, sym in enumerate(returns_df.columns):
        result += f"- {sym}: {asset_vols[i]*100:.1f}%\n"

    result += "\n**Why Risk Parity?**\n"
    result += "- Equal risk contribution provides true diversification\n"
    result += "- Avoids concentration in high-volatility assets\n"
    result += "- More stable in different market regimes\n"
    result += "- Bridgewater's All Weather strategy uses this approach\n"

    return result


@tool
def black_litterman_portfolio_tool(
    symbols: str,
    views: str,
    tau: float = 0.025,
) -> str:
    """Optimize portfolio using Black-Litterman model with investor views.

    Black-Litterman combines market equilibrium returns with investor views
    to create more stable and intuitive portfolio allocations.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL")
        views: Investor views in format "SYMBOL:VIEW_RETURN:CONFIDENCE" (e.g.,
              "AAPL:0.15:0.8,MSFT:0.10:0.6" means AAPL expected to return 15%
              with 80% confidence, MSFT 10% with 60% confidence)
        tau: Scaling factor for uncertainty (default: 0.025)

    Returns:
        String with Black-Litterman portfolio weights and expected returns.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    # Parse views
    view_dict = {}
    for item in views.split(","):
        parts = item.strip().split(":")
        if len(parts) >= 2:
            sym = parts[0].strip().upper()
            expected_return = float(parts[1].strip())
            confidence = float(parts[2].strip()) if len(parts) > 2 else 0.5
            view_dict[sym] = {"return": expected_return, "confidence": confidence}

    if not view_dict:
        return "Error: Invalid views format. Use 'SYMBOL:RETURN:CONFIDENCE'"

    data_pipeline, _ = get_pipeline()

    async def fetch():
        prices = await data_pipeline.get_historical_prices_batch(symbol_list, limit=252)
        profiles = await data_pipeline.get_company_profiles_batch(symbol_list)
        await data_pipeline._client.close()
        return prices, profiles

    prices_dict, profiles = run_async(fetch())

    # Build returns and covariance matrix
    import pandas as pd
    returns_data = {}
    for symbol, prices in prices_dict.items():
        if prices:
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            returns = np.diff(np.log(closes))
            returns_data[symbol] = returns

    if len(returns_data) < 2:
        return "Error: Need at least 2 symbols with valid data."

    # Align returns
    min_len = min(len(r) for r in returns_data.values())
    for sym in returns_data:
        returns_data[sym] = returns_data[sym][-min_len:]

    returns_df = pd.DataFrame(returns_data)
    cov_matrix = returns_df.cov().values * 252

    # Get market caps for equilibrium weights
    market_caps = {}
    for p in profiles:
        if p.symbol in returns_df.columns:
            market_caps[p.symbol] = p.mkt_cap or 1e10

    # Calculate market weights
    total_cap = sum(market_caps.values())
    market_weights = np.array([
        market_caps.get(sym, total_cap / len(symbol_list)) / total_cap
        for sym in returns_df.columns
    ])

    # Build P matrix (picking matrix) and Q vector (views)
    view_symbols = [s for s in view_dict.keys() if s in returns_df.columns]
    n_assets = len(returns_df.columns)
    n_views = len(view_symbols)

    if n_views == 0:
        return "Error: No valid views found for the provided symbols."

    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    omega_diag = []

    for i, sym in enumerate(view_symbols):
        col_idx = list(returns_df.columns).index(sym)
        P[i, col_idx] = 1.0
        Q[i] = view_dict[sym]["return"]
        # Uncertainty inversely proportional to confidence
        confidence = view_dict[sym]["confidence"]
        uncertainty = (1 - confidence) * 0.1  # Scale uncertainty
        omega_diag.append(max(uncertainty, 0.001))

    # Run Black-Litterman
    result_data = PortfolioManagement.black_litterman(
        market_weights=market_weights,
        cov_matrix=cov_matrix,
        P=P,
        Q=Q,
        tau=tau,
        omega=np.diag(omega_diag),
    )

    # Build result string
    result = "**Black-Litterman Portfolio Optimization**\n\n"
    result += "**Methodology:**\n"
    result += "Black-Litterman starts with market equilibrium returns and adjusts them\n"
    result += "based on your views, weighted by your confidence level.\n\n"

    result += "**Formula:**\n"
    result += "```\n"
    result += "E[R] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]\n"
    result += "\n"
    result += "Where:\n"
    result += "  π = equilibrium excess returns (from CAPM)\n"
    result += "  Σ = covariance matrix\n"
    result += "  P = picking matrix (which assets have views)\n"
    result += "  Q = view returns\n"
    result += "  Ω = uncertainty in views\n"
    result += "  τ = scaling factor (typically 0.025)\n"
    result += "```\n\n"

    result += "**Your Views:**\n"
    for sym in view_symbols:
        v = view_dict[sym]
        result += f"- {sym}: {v['return']*100:.1f}% expected return ({v['confidence']*100:.0f}% confidence)\n"

    result += "\n**Equilibrium vs Black-Litterman Returns:**\n"
    result += "```\n"
    result += f"{'Symbol':<8} {'Market Eq.':<12} {'BL Adjusted':<12} {'Difference':<12}\n"
    result += "-" * 44 + "\n"
    for i, sym in enumerate(returns_df.columns):
        eq_ret = result_data.equilibrium_returns[i] * 100
        bl_ret = result_data.posterior_returns[i] * 100
        diff = bl_ret - eq_ret
        result += f"{sym:<8} {eq_ret:>10.2f}% {bl_ret:>10.2f}% {diff:>+10.2f}%\n"
    result += "```\n\n"

    result += "**Optimal Weights:**\n"
    result += "```\n"
    result += f"{'Symbol':<8} {'Market Wt.':<12} {'BL Weight':<12} {'Change':<12}\n"
    result += "-" * 44 + "\n"
    for i, sym in enumerate(returns_df.columns):
        mkt_wt = market_weights[i] * 100
        bl_wt = result_data.optimal_weights[i] * 100
        change = bl_wt - mkt_wt
        result += f"{sym:<8} {mkt_wt:>10.1f}% {bl_wt:>10.1f}% {change:>+10.1f}%\n"
    result += "```\n\n"

    result += "**Why Black-Litterman?**\n"
    result += "- Combines market consensus with your own views\n"
    result += "- Produces more intuitive, diversified portfolios\n"
    result += "- Handles uncertainty in forecasts explicitly\n"
    result += "- Views with low confidence have less impact\n"

    return result


@tool
def kelly_criterion_tool(
    symbols: str,
    fraction: float = 0.5,
    risk_free_rate: float = 0.05,
) -> str:
    """Calculate optimal position sizes using Kelly Criterion.

    Kelly Criterion determines the optimal fraction of capital to bet to maximize
    long-term growth rate. Full Kelly is often too aggressive, so a fractional
    Kelly (e.g., 0.5) is commonly used.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL")
        fraction: Kelly fraction to use (0.5 = half-Kelly, recommended for stability)
        risk_free_rate: Risk-free rate (default: 0.05 = 5%)

    Returns:
        String with Kelly optimal weights and growth rate analysis.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    data_pipeline, _ = get_pipeline()

    async def fetch():
        prices = await data_pipeline.get_historical_prices_batch(symbol_list, limit=252)
        await data_pipeline._client.close()
        return prices

    prices_dict = run_async(fetch())

    # Build returns
    import pandas as pd
    returns_data = {}
    for symbol, prices in prices_dict.items():
        if prices:
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            returns = np.diff(np.log(closes))
            returns_data[symbol] = returns

    if len(returns_data) < 1:
        return "Error: Need at least 1 symbol with valid data."

    min_len = min(len(r) for r in returns_data.values())
    for sym in returns_data:
        returns_data[sym] = returns_data[sym][-min_len:]

    returns_df = pd.DataFrame(returns_data)

    # Calculate expected returns and covariance
    expected_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252

    # Run Kelly Criterion
    result_data = PortfolioManagement.kelly_criterion(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        fraction=fraction,
    )

    # Build result
    result = "**Kelly Criterion Portfolio Sizing**\n\n"
    result += "**Methodology:**\n"
    result += "Kelly Criterion maximizes the expected logarithmic growth rate of capital.\n"
    result += f"Using {fraction:.0%} Kelly (fractional Kelly for stability).\n\n"

    result += "**Formula:**\n"
    result += "```\n"
    result += "f* = Σ⁻¹ × (μ - rᶠ)\n"
    result += "\n"
    result += "Where:\n"
    result += "  f* = optimal fraction of capital for each asset\n"
    result += "  Σ⁻¹ = inverse covariance matrix\n"
    result += "  μ = expected returns vector\n"
    result += "  rᶠ = risk-free rate\n"
    result += "\n"
    result += "Kelly Fraction Applied:\n"
    result += f"  f_actual = {fraction} × f*\n"
    result += "```\n\n"

    result += "**Full Kelly vs Fractional Kelly Weights:**\n"
    result += "```\n"
    result += f"{'Symbol':<8} {'Full Kelly':<14} {f'{fraction:.0%} Kelly':<14} {'Exp.Return':<12}\n"
    result += "-" * 48 + "\n"
    for i, sym in enumerate(returns_df.columns):
        full_kelly = result_data["full_kelly_weights"][i] * 100
        frac_kelly = result_data["fractional_kelly_weights"][i] * 100
        exp_ret = expected_returns[i] * 100
        result += f"{sym:<8} {full_kelly:>12.1f}% {frac_kelly:>12.1f}% {exp_ret:>10.1f}%\n"
    result += "```\n\n"

    result += "**Growth Rate Analysis:**\n"
    result += f"- Expected Growth Rate (Full Kelly): {result_data['expected_growth_rate']*100:.2f}%\n"
    result += f"- Estimated Volatility: {np.sqrt(result_data['expected_growth_rate'])*100:.2f}% (approx)\n\n"

    # Cash allocation if weights < 100%
    total_weight = sum(result_data["fractional_kelly_weights"])
    if total_weight < 1.0:
        result += f"**Cash Allocation:** {(1-total_weight)*100:.1f}%\n"
        result += "(Remaining capital held in risk-free asset)\n\n"
    elif total_weight > 1.0:
        result += f"**Leverage Required:** {total_weight:.1f}x\n"
        result += "(Full Kelly often requires leverage - use with caution)\n\n"

    result += "**Why Use Fractional Kelly?**\n"
    result += "- Full Kelly maximizes growth but has high volatility\n"
    result += f"- {fraction:.0%} Kelly reduces volatility significantly\n"
    result += "- Only sacrifices a small amount of expected growth\n"
    result += "- More robust to estimation errors in returns\n"
    result += "- Warren Buffett: 'Rule #1: Never lose money'\n"

    return result


@tool
def monte_carlo_wealth_tool(
    initial_wealth: float,
    expected_return: float,
    volatility: float,
    years: int = 30,
    annual_contribution: float = 0,
    target_wealth: float = 0,
) -> str:
    """Run Monte Carlo simulation for wealth projection.

    Simulate thousands of potential wealth paths to understand the range of
    possible outcomes and probability of reaching financial goals.

    Args:
        initial_wealth: Starting portfolio value (e.g., 100000)
        expected_return: Expected annual return (e.g., 0.08 for 8%)
        volatility: Annual volatility (e.g., 0.15 for 15%)
        years: Investment horizon in years (default: 30)
        annual_contribution: Amount added each year (default: 0)
        target_wealth: Target wealth to calculate probability of reaching

    Returns:
        String with wealth simulation results and probabilities.
    """
    # Run simulation
    result_data = PortfolioManagement.monte_carlo_wealth_simulation(
        initial_wealth=initial_wealth,
        expected_return=expected_return,
        volatility=volatility,
        years=years,
        annual_contribution=annual_contribution,
        n_simulations=10000,
    )

    # Build result
    result = "**Monte Carlo Wealth Simulation**\n\n"
    result += "**Parameters:**\n"
    result += f"- Initial Wealth: ${initial_wealth:,.0f}\n"
    result += f"- Expected Annual Return: {expected_return*100:.1f}%\n"
    result += f"- Annual Volatility: {volatility*100:.1f}%\n"
    result += f"- Investment Horizon: {years} years\n"
    if annual_contribution > 0:
        result += f"- Annual Contribution: ${annual_contribution:,.0f}\n"
    result += f"- Simulations: 10,000\n\n"

    result += "**Methodology:**\n"
    result += "```\n"
    result += "W(t+1) = W(t) × exp[(μ - σ²/2)Δt + σ√Δt × Z]\n"
    result += "\n"
    result += "Where:\n"
    result += "  W(t) = wealth at time t\n"
    result += "  μ = expected return\n"
    result += "  σ = volatility\n"
    result += "  Z = standard normal random variable\n"
    result += "  Δt = time step (1 year)\n"
    result += "```\n\n"

    result += "**Wealth Projection (End of Period):**\n"
    result += "```\n"
    result += f"{'Percentile':<15} {'Wealth':<20} {'Total Return':<15}\n"
    result += "-" * 50 + "\n"

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        key = f"p{p}"
        if key in result_data["final_wealth_percentiles"]:
            wealth = result_data["final_wealth_percentiles"][key]
            total_return = (wealth / initial_wealth - 1) * 100
            result += f"{p}th percentile  ${wealth:>15,.0f}  {total_return:>+12.1f}%\n"

    result += "```\n\n"

    result += "**Key Statistics:**\n"
    result += f"- Mean Final Wealth: ${result_data['mean_final_wealth']:,.0f}\n"
    result += f"- Median Final Wealth: ${result_data['median_final_wealth']:,.0f}\n"
    result += f"- Std Dev of Final Wealth: ${result_data['std_final_wealth']:,.0f}\n"
    result += f"- Probability of Loss: {result_data['probability_of_loss']*100:.1f}%\n\n"

    if target_wealth > 0:
        # Calculate probability of reaching target
        prob_target = result_data.get("probability_above_target", 0)
        if prob_target == 0:
            # Estimate from percentiles
            final_wealths = result_data.get("final_wealths", [])
            if final_wealths:
                prob_target = np.mean(np.array(final_wealths) >= target_wealth)

        result += f"**Goal Analysis:**\n"
        result += f"- Target Wealth: ${target_wealth:,.0f}\n"
        result += f"- Probability of Reaching Target: ~{prob_target*100:.1f}%\n\n"

    result += "**Interpretation:**\n"
    median = result_data['median_final_wealth']
    result += f"- 50% chance of having more than ${median:,.0f}\n"
    p10 = result_data['final_wealth_percentiles'].get('p10', median * 0.5)
    p90 = result_data['final_wealth_percentiles'].get('p90', median * 2)
    result += f"- 80% confidence range: ${p10:,.0f} to ${p90:,.0f}\n"
    loss_prob = result_data['probability_of_loss'] * 100
    if loss_prob > 10:
        result += f"- ⚠️ {loss_prob:.1f}% chance of losing money - consider more conservative allocation\n"
    else:
        result += f"- ✅ Only {loss_prob:.1f}% chance of losing money over {years} years\n"

    return result


@tool
def advanced_portfolio_optimization_tool(
    symbols: str,
    method: str = "all",
    risk_free_rate: float = 0.05,
) -> str:
    """Compare multiple portfolio optimization methods side-by-side.

    Runs Mean-Variance, Risk Parity, Maximum Diversification, and Kelly Criterion
    optimizations to compare different allocation strategies.

    Args:
        symbols: Comma-separated stock symbols (e.g., "AAPL,MSFT,GOOGL,BND")
        method: Optimization method - "all", "mean_variance", "risk_parity",
               "max_diversification", or "kelly" (default: "all")
        risk_free_rate: Risk-free rate for Sharpe calculation (default: 0.05)

    Returns:
        String comparing different optimization approaches.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    data_pipeline, _ = get_pipeline()

    async def fetch():
        prices = await data_pipeline.get_historical_prices_batch(symbol_list, limit=252)
        await data_pipeline._client.close()
        return prices

    prices_dict = run_async(fetch())

    # Build returns
    import pandas as pd
    returns_data = {}
    for symbol, prices in prices_dict.items():
        if prices:
            closes = [p.close for p in sorted(prices, key=lambda x: x.date)]
            returns = np.diff(np.log(closes))
            returns_data[symbol] = returns

    if len(returns_data) < 2:
        return "Error: Need at least 2 symbols with valid data."

    min_len = min(len(r) for r in returns_data.values())
    for sym in returns_data:
        returns_data[sym] = returns_data[sym][-min_len:]

    returns_df = pd.DataFrame(returns_data)
    expected_returns = returns_df.mean().values * 252
    cov_matrix = returns_df.cov().values * 252
    n_assets = len(returns_df.columns)

    results = {}

    # Equal Weight (baseline)
    equal_weights = np.ones(n_assets) / n_assets
    eq_ret = np.dot(equal_weights, expected_returns)
    eq_vol = np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights)))
    eq_sharpe = (eq_ret - risk_free_rate) / eq_vol if eq_vol > 0 else 0
    results["Equal Weight"] = {
        "weights": equal_weights,
        "return": eq_ret,
        "volatility": eq_vol,
        "sharpe": eq_sharpe,
    }

    # Mean-Variance (Max Sharpe)
    if method in ["all", "mean_variance"]:
        try:
            mv_result = PortfolioManagement.mean_variance_optimization(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                risk_free_rate=risk_free_rate,
            )
            # Find max sharpe portfolio
            max_sharpe_idx = np.argmax(mv_result.sharpe_ratios)
            mv_weights = mv_result.efficient_weights[max_sharpe_idx]
            mv_ret = mv_result.efficient_returns[max_sharpe_idx]
            mv_vol = mv_result.efficient_volatilities[max_sharpe_idx]
            mv_sharpe = mv_result.sharpe_ratios[max_sharpe_idx]
            results["Mean-Variance (Max Sharpe)"] = {
                "weights": mv_weights,
                "return": mv_ret,
                "volatility": mv_vol,
                "sharpe": mv_sharpe,
            }
        except Exception:
            pass

    # Risk Parity
    if method in ["all", "risk_parity"]:
        try:
            rp_result = PortfolioManagement.risk_parity_portfolio(cov_matrix=cov_matrix)
            rp_weights = rp_result.weights
            rp_ret = np.dot(rp_weights, expected_returns)
            rp_vol = rp_result.portfolio_volatility
            rp_sharpe = (rp_ret - risk_free_rate) / rp_vol if rp_vol > 0 else 0
            results["Risk Parity"] = {
                "weights": rp_weights,
                "return": rp_ret,
                "volatility": rp_vol,
                "sharpe": rp_sharpe,
            }
        except Exception:
            pass

    # Maximum Diversification
    if method in ["all", "max_diversification"]:
        try:
            md_result = PortfolioManagement.maximum_diversification_portfolio(
                cov_matrix=cov_matrix
            )
            md_weights = md_result["weights"]
            md_ret = np.dot(md_weights, expected_returns)
            md_vol = md_result["portfolio_volatility"]
            md_sharpe = (md_ret - risk_free_rate) / md_vol if md_vol > 0 else 0
            results["Max Diversification"] = {
                "weights": md_weights,
                "return": md_ret,
                "volatility": md_vol,
                "sharpe": md_sharpe,
                "div_ratio": md_result["diversification_ratio"],
            }
        except Exception:
            pass

    # Kelly Criterion (Half Kelly)
    if method in ["all", "kelly"]:
        try:
            kelly_result = PortfolioManagement.kelly_criterion(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                risk_free_rate=risk_free_rate,
                fraction=0.5,
            )
            kelly_weights = np.array(kelly_result["fractional_kelly_weights"])
            # Normalize if sum > 1 for comparison
            if kelly_weights.sum() > 1:
                kelly_weights_norm = kelly_weights / kelly_weights.sum()
            else:
                kelly_weights_norm = kelly_weights
            kelly_ret = np.dot(kelly_weights_norm, expected_returns)
            kelly_vol = np.sqrt(np.dot(kelly_weights_norm, np.dot(cov_matrix, kelly_weights_norm)))
            kelly_sharpe = (kelly_ret - risk_free_rate) / kelly_vol if kelly_vol > 0 else 0
            results["Half-Kelly"] = {
                "weights": kelly_weights_norm,
                "return": kelly_ret,
                "volatility": kelly_vol,
                "sharpe": kelly_sharpe,
            }
        except Exception:
            pass

    # Build result string
    result = "**Advanced Portfolio Optimization Comparison**\n\n"
    result += f"Symbols: {', '.join(returns_df.columns)}\n"
    result += f"Risk-Free Rate: {risk_free_rate*100:.1f}%\n\n"

    # Weights comparison
    result += "**Optimal Weights by Method:**\n"
    result += "```\n"
    header = f"{'Symbol':<8}"
    for method_name in results.keys():
        short_name = method_name[:12]
        header += f"{short_name:>14}"
    result += header + "\n"
    result += "-" * (8 + 14 * len(results)) + "\n"

    for i, sym in enumerate(returns_df.columns):
        row = f"{sym:<8}"
        for method_name, data in results.items():
            weight = data["weights"][i] * 100
            row += f"{weight:>13.1f}%"
        result += row + "\n"
    result += "```\n\n"

    # Performance comparison
    result += "**Expected Performance:**\n"
    result += "```\n"
    result += f"{'Method':<24} {'Return':<10} {'Vol':<10} {'Sharpe':<10}\n"
    result += "-" * 54 + "\n"
    for method_name, data in results.items():
        ret = data["return"] * 100
        vol = data["volatility"] * 100
        sharpe = data["sharpe"]
        result += f"{method_name:<24} {ret:>8.1f}% {vol:>8.1f}% {sharpe:>8.2f}\n"
    result += "```\n\n"

    # Find best
    best_sharpe = max(results.items(), key=lambda x: x[1]["sharpe"])
    lowest_vol = min(results.items(), key=lambda x: x[1]["volatility"])
    highest_ret = max(results.items(), key=lambda x: x[1]["return"])

    result += "**Recommendations:**\n"
    result += f"- **Best Risk-Adjusted Return:** {best_sharpe[0]} (Sharpe: {best_sharpe[1]['sharpe']:.2f})\n"
    result += f"- **Lowest Volatility:** {lowest_vol[0]} ({lowest_vol[1]['volatility']*100:.1f}%)\n"
    result += f"- **Highest Expected Return:** {highest_ret[0]} ({highest_ret[1]['return']*100:.1f}%)\n\n"

    result += "**Method Descriptions:**\n"
    result += "- **Equal Weight:** Simple 1/N allocation - surprisingly robust\n"
    result += "- **Mean-Variance:** Markowitz optimization maximizing Sharpe ratio\n"
    result += "- **Risk Parity:** Equal risk contribution from each asset\n"
    result += "- **Max Diversification:** Maximizes diversification ratio\n"
    result += "- **Half-Kelly:** Growth-optimal sizing with reduced volatility\n"

    return result


@tool
def glide_path_tool(
    current_age: int,
    retirement_age: int = 65,
    risk_profile: str = "moderate",
) -> str:
    """Generate a lifecycle glide path for retirement investing.

    Creates an age-appropriate asset allocation that shifts from stocks to bonds
    as you approach retirement.

    Args:
        current_age: Current age of the investor
        retirement_age: Target retirement age (default: 65)
        risk_profile: "conservative", "moderate", or "aggressive"

    Returns:
        String with glide path allocations by age.
    """
    # Generate glide path
    glide_path = PortfolioManagement.generate_glide_path(
        current_age=current_age,
        retirement_age=retirement_age,
        risk_profile=risk_profile,
    )

    result = "**Lifecycle Glide Path**\n\n"
    result += f"**Profile:** {risk_profile.title()}\n"
    result += f"**Current Age:** {current_age}\n"
    result += f"**Retirement Age:** {retirement_age}\n"
    result += f"**Years to Retirement:** {max(0, retirement_age - current_age)}\n\n"

    result += "**Methodology:**\n"
    result += "Glide paths gradually shift from growth assets (stocks) to income assets\n"
    result += "(bonds) as you approach and enter retirement. This reduces sequence-of-returns\n"
    result += "risk near and during retirement.\n\n"

    result += "**Your Allocation Over Time:**\n"
    result += "```\n"
    result += f"{'Age':<6} {'Stocks':<10} {'Bonds':<10} {'Cash':<10}\n"
    result += "-" * 36 + "\n"

    # Show key ages
    ages_to_show = list(range(current_age, min(current_age + 50, 95), 5))
    if retirement_age not in ages_to_show:
        ages_to_show.append(retirement_age)
    ages_to_show = sorted(set(ages_to_show))

    for age in ages_to_show:
        if age in glide_path:
            alloc = glide_path[age]
            stocks = alloc.get("equity", 0) * 100
            bonds = alloc.get("fixed_income", 0) * 100
            cash = alloc.get("cash", 0) * 100
            marker = " <-- YOU" if age == current_age else (" <-- RETIRE" if age == retirement_age else "")
            result += f"{age:<6} {stocks:>8.0f}% {bonds:>8.0f}% {cash:>8.0f}%{marker}\n"

    result += "```\n\n"

    # Current allocation
    if current_age in glide_path:
        current_alloc = glide_path[current_age]
        result += "**Your Current Recommended Allocation:**\n"
        result += f"- Stocks/Equity: {current_alloc.get('equity', 0)*100:.0f}%\n"
        result += f"- Bonds/Fixed Income: {current_alloc.get('fixed_income', 0)*100:.0f}%\n"
        result += f"- Cash/Money Market: {current_alloc.get('cash', 0)*100:.0f}%\n\n"

    result += "**Risk Profile Differences:**\n"
    result += "- **Conservative:** Higher bond allocation, earlier de-risking\n"
    result += "- **Moderate:** Balanced approach (most common)\n"
    result += "- **Aggressive:** More equity, later de-risking for higher growth potential\n\n"

    result += "**Implementation Suggestions:**\n"
    result += "- **Stocks:** Total market index, S&P 500 (VTI, SPY, VOO)\n"
    result += "- **Bonds:** Aggregate bond index (BND, AGG)\n"
    result += "- **Cash:** Money market, Treasury bills\n"
    result += "- Consider target-date funds that automate this glide path\n"

    return result
