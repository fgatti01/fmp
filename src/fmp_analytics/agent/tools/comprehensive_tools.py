"""Comprehensive analysis tools.

Tools for in-depth equity and sector analysis with fair value calculations,
earnings surprises, and correlation analysis.
"""

from fmp_analytics.agent.tools._common import (
    date,
    timedelta,
    np,
    pd,
    tool,
    get_pipeline,
    run_async,
)

@tool
def earnings_surprise_analysis_tool(
    sector: str = "",
    limit: int = 20,
) -> str:
    """Analyze recent earnings surprises and their market impact.

    Use this tool to find stocks with significant earnings beats or misses
    and understand how the market reacted.

    Args:
        sector: Optional sector filter (leave empty for all sectors)
        limit: Number of companies to analyze (default: 20)

    Returns:
        String with earnings surprise analysis and market reactions.
    """
    data_pipeline, _ = get_pipeline()

    async def fetch():
        # Get earnings calendar (recent)
        to_date = date.today()
        from_date = to_date - timedelta(days=30)

        earnings = await data_pipeline.get_earnings_calendar(str(from_date), str(to_date))

        # Get quotes for companies with earnings
        if not earnings.empty:
            symbols = earnings["symbol"].head(limit).tolist()
            quotes = await data_pipeline.get_quotes_batch(symbols)
        else:
            quotes = []

        await data_pipeline._client.close()
        return earnings, quotes

    earnings_df, quotes = run_async(fetch())

    if earnings_df.empty:
        return "No recent earnings data available."

    result = "**Recent Earnings Surprises Analysis**\n\n"

    quote_lookup = {q.symbol: q for q in quotes}

    beats = []
    misses = []
    in_line = []

    for _, row in earnings_df.head(limit).iterrows():
        symbol = row.get("symbol", "")
        actual = row.get("eps", 0) or 0
        estimate = row.get("epsEstimated", 0) or 0
        report_date = row.get("date", "")

        if not estimate or estimate == 0:
            continue

        surprise_pct = ((actual - estimate) / abs(estimate)) * 100

        quote = quote_lookup.get(symbol)
        price_change = quote.changes_percentage if quote else 0

        entry = {
            "symbol": symbol,
            "actual": actual,
            "estimate": estimate,
            "surprise": surprise_pct,
            "date": report_date,
            "price_change": price_change,
            "price": quote.price if quote else 0,
        }

        if surprise_pct > 5:
            beats.append(entry)
        elif surprise_pct < -5:
            misses.append(entry)
        else:
            in_line.append(entry)

    # Sort by surprise magnitude
    beats.sort(key=lambda x: -x["surprise"])
    misses.sort(key=lambda x: x["surprise"])

    if beats:
        result += "**📈 Significant Earnings Beats (>5% surprise):**\n\n"
        for b in beats[:7]:
            emoji = "✅" if b["price_change"] > 0 else "⚠️"
            result += f"**{b['symbol']}** - {b['date']}\n"
            result += f"- EPS: ${b['actual']:.2f} vs ${b['estimate']:.2f} est\n"
            result += f"- Surprise: **+{b['surprise']:.1f}%**\n"
            result += f"- Stock Reaction: {emoji} {b['price_change']:+.1f}%\n"
            if b["price_change"] < 0 and b["surprise"] > 10:
                result += f"  *⚠️ Beat but sold off - possible guidance concerns or 'sell the news'*\n"
            elif b["price_change"] > b["surprise"]:
                result += f"  *Stock reaction exceeded surprise - positive forward guidance likely*\n"
            result += "\n"

    if misses:
        result += "**📉 Significant Earnings Misses (<-5% surprise):**\n\n"
        for m in misses[:7]:
            emoji = "❌" if m["price_change"] < 0 else "🤔"
            result += f"**{m['symbol']}** - {m['date']}\n"
            result += f"- EPS: ${m['actual']:.2f} vs ${m['estimate']:.2f} est\n"
            result += f"- Surprise: **{m['surprise']:.1f}%**\n"
            result += f"- Stock Reaction: {emoji} {m['price_change']:+.1f}%\n"
            if m["price_change"] > 0 and m["surprise"] < -10:
                result += f"  *🤔 Missed but rallied - likely positive guidance or low expectations*\n"
            elif m["price_change"] < m["surprise"]:
                result += f"  *Stock fell more than miss - additional concerns beyond EPS*\n"
            result += "\n"

    # Summary statistics
    all_surprises = beats + misses + in_line
    if all_surprises:
        avg_surprise = sum(e["surprise"] for e in all_surprises) / len(all_surprises)
        beat_rate = len(beats) / len(all_surprises) * 100

        result += "**Summary Statistics:**\n"
        result += f"- Companies analyzed: {len(all_surprises)}\n"
        result += f"- Beat rate: {beat_rate:.0f}%\n"
        result += f"- Average surprise: {avg_surprise:+.1f}%\n"
        result += f"- Significant beats: {len(beats)}\n"
        result += f"- Significant misses: {len(misses)}\n"

    return result


@tool
def comprehensive_equity_analysis_tool(
    symbol: str,
) -> str:
    """Perform complete equity analysis with fundamental and quantitative metrics.

    This tool gathers ALL available data for an equity and performs:
    - Fundamental analysis (DCF, DDM, multiples, Graham number)
    - Quantitative analysis (risk metrics, technical indicators, momentum)
    - Fair value calculation using multiple methods
    - Comparison of current price vs calculated fair values
    - Buy/Sell/Hold recommendation based on the analysis

    Use this tool when you need a complete picture of an equity's value and prospects.

    Args:
        symbol: Stock symbol (e.g., "AAPL")

    Returns:
        String with comprehensive equity analysis including fair value vs market price.
    """
    data_pipeline, analysis_pipeline = get_pipeline()

    async def fetch():
        sym = symbol.upper()
        to_date = date.today()
        from_date = to_date - timedelta(days=365)

        # Gather ALL data
        quote_data = await data_pipeline._market_data.get_quote(sym)
        quote = quote_data[0] if quote_data else None

        profile_data = await data_pipeline._company_info.get_profile(sym)
        profile = profile_data[0] if profile_data else None

        # Financial statements
        income_stmt = await data_pipeline._financials.get_income_statement(sym, "annual", limit=5)
        balance_sheet = await data_pipeline._financials.get_balance_sheet(sym, "annual", limit=5)
        cash_flow = await data_pipeline._financials.get_cash_flow_statement(sym, "annual", limit=5)

        # Ratios and metrics
        ratios_df = await data_pipeline.get_financial_ratios(sym, "annual", limit=3)
        key_metrics = await data_pipeline._financials.get_key_metrics(sym, "annual", limit=3)

        # Historical prices for technical analysis
        hist = await data_pipeline.get_historical_prices(sym, str(from_date), str(to_date))

        # DCF value from FMP
        dcf_data = await data_pipeline._company_info.get_dcf(sym)

        # Earnings surprises
        earnings = await data_pipeline._financials.get_earnings_surprises(sym)

        # Analyst estimates
        try:
            estimates = await data_pipeline._financials.get_analyst_estimates(sym, "annual")
        except Exception:
            estimates = []

        # Analyst recommendations
        try:
            recommendations = await data_pipeline._company_info.get_analyst_recommendations(sym)
        except Exception:
            recommendations = []

        # Sector peers for comparison
        if profile:
            peers_data = await data_pipeline._stock_list.get_stock_screener(
                sector=profile.get("sector"),
                is_actively_trading=True,
                limit=20,
            )
            peer_symbols = [p["symbol"] for p in peers_data if p["symbol"] != sym][:10]
            peer_quotes = await data_pipeline.get_quotes_batch(peer_symbols)
        else:
            peer_quotes = []

        await data_pipeline._client.close()
        return (quote, profile, income_stmt, balance_sheet, cash_flow, ratios_df,
                key_metrics, hist, dcf_data, earnings, estimates, recommendations, peer_quotes)

    (quote, profile, income_stmt, balance_sheet, cash_flow, ratios_df,
     key_metrics, hist, dcf_data, earnings, estimates, recommendations, peer_quotes) = run_async(fetch())

    if not quote or not profile:
        return f"Unable to fetch data for {symbol}"

    import numpy as np

    result = f"# Comprehensive Equity Analysis: {symbol.upper()}\n\n"

    # ===== COMPANY OVERVIEW =====
    result += "## 1. Company Overview\n\n"
    result += f"**{profile.get('companyName', symbol)}** ({symbol.upper()})\n"
    result += f"- Sector: {profile.get('sector', 'N/A')}\n"
    result += f"- Industry: {profile.get('industry', 'N/A')}\n"
    result += f"- Market Cap: ${profile.get('mktCap', 0):,.0f}\n"
    result += f"- Employees: {profile.get('fullTimeEmployees', 'N/A'):,}\n"
    result += f"- Exchange: {profile.get('exchange', 'N/A')}\n\n"

    current_price = quote.get("price", 0)
    result += f"**Current Price: ${current_price:.2f}**\n"
    result += f"- Change Today: {quote.get('change', 0):+.2f} ({quote.get('changesPercentage', 0):+.2f}%)\n"
    result += f"- 52-Week Range: ${quote.get('yearLow', 0):.2f} - ${quote.get('yearHigh', 0):.2f}\n"
    result += f"- Volume: {quote.get('volume', 0):,} (Avg: {quote.get('avgVolume', 0):,})\n\n"

    # ===== FUNDAMENTAL ANALYSIS =====
    result += "## 2. Fundamental Analysis\n\n"

    # Financial Statement Analysis
    result += "### 2.1 Financial Performance\n\n"

    if income_stmt:
        latest_income = income_stmt[0] if income_stmt else {}
        prev_income = income_stmt[1] if len(income_stmt) > 1 else {}

        revenue = latest_income.get("revenue", 0)
        net_income = latest_income.get("netIncome", 0)
        gross_profit = latest_income.get("grossProfit", 0)
        operating_income = latest_income.get("operatingIncome", 0)
        eps = latest_income.get("eps", 0)

        prev_revenue = prev_income.get("revenue", 1)
        revenue_growth = ((revenue / prev_revenue) - 1) * 100 if prev_revenue else 0

        result += "**Income Statement (Latest Year):**\n"
        result += f"- Revenue: ${revenue:,.0f}\n"
        result += f"- Revenue Growth: {revenue_growth:+.1f}% YoY\n"
        result += f"- Gross Profit: ${gross_profit:,.0f} ({gross_profit/revenue*100:.1f}% margin)\n"
        result += f"- Operating Income: ${operating_income:,.0f} ({operating_income/revenue*100:.1f}% margin)\n"
        result += f"- Net Income: ${net_income:,.0f} ({net_income/revenue*100:.1f}% margin)\n"
        result += f"- EPS: ${eps:.2f}\n\n"

    if balance_sheet:
        latest_bs = balance_sheet[0] if balance_sheet else {}

        total_assets = latest_bs.get("totalAssets", 0)
        total_debt = latest_bs.get("totalDebt", 0)
        total_equity = latest_bs.get("totalStockholdersEquity", 0)
        cash = latest_bs.get("cashAndCashEquivalents", 0)

        result += "**Balance Sheet:**\n"
        result += f"- Total Assets: ${total_assets:,.0f}\n"
        result += f"- Cash & Equivalents: ${cash:,.0f}\n"
        result += f"- Total Debt: ${total_debt:,.0f}\n"
        result += f"- Shareholders' Equity: ${total_equity:,.0f}\n"
        result += f"- Debt/Equity Ratio: {total_debt/total_equity:.2f}\n\n" if total_equity else ""

    if cash_flow:
        latest_cf = cash_flow[0] if cash_flow else {}

        operating_cf = latest_cf.get("operatingCashFlow", 0)
        fcf = latest_cf.get("freeCashFlow", 0)
        capex = latest_cf.get("capitalExpenditure", 0)
        dividends = latest_cf.get("dividendsPaid", 0)

        result += "**Cash Flow:**\n"
        result += f"- Operating Cash Flow: ${operating_cf:,.0f}\n"
        result += f"- Free Cash Flow: ${fcf:,.0f}\n"
        result += f"- CapEx: ${capex:,.0f}\n"
        result += f"- Dividends Paid: ${abs(dividends):,.0f}\n\n"

    # Key Ratios
    result += "### 2.2 Key Financial Ratios\n\n"

    if not ratios_df.empty:
        r = ratios_df.iloc[0]

        result += "**Profitability:**\n"
        result += f"- Gross Margin: {r.get('grossProfitMargin', 0)*100:.1f}%\n"
        result += f"- Operating Margin: {r.get('operatingProfitMargin', 0)*100:.1f}%\n"
        result += f"- Net Margin: {r.get('netProfitMargin', 0)*100:.1f}%\n"
        result += f"- ROE: {r.get('returnOnEquity', 0)*100:.1f}%\n"
        result += f"- ROA: {r.get('returnOnAssets', 0)*100:.1f}%\n"
        result += f"- ROIC: {r.get('returnOnCapitalEmployed', 0)*100:.1f}%\n\n"

        result += "**Liquidity:**\n"
        result += f"- Current Ratio: {r.get('currentRatio', 0):.2f}\n"
        result += f"- Quick Ratio: {r.get('quickRatio', 0):.2f}\n"
        result += f"- Cash Ratio: {r.get('cashRatio', 0):.2f}\n\n"

        result += "**Valuation Multiples:**\n"
        pe_ratio = r.get('priceEarningsRatio', 0) or quote.get('pe', 0)
        pb_ratio = r.get('priceToBookRatio', 0)
        ps_ratio = r.get('priceToSalesRatio', 0)
        ev_ebitda = r.get('enterpriseValueMultiple', 0)
        peg_ratio = r.get('priceEarningsToGrowthRatio', 0)

        result += f"- P/E Ratio: {pe_ratio:.2f}\n"
        result += f"- P/B Ratio: {pb_ratio:.2f}\n"
        result += f"- P/S Ratio: {ps_ratio:.2f}\n"
        result += f"- EV/EBITDA: {ev_ebitda:.2f}\n"
        result += f"- PEG Ratio: {peg_ratio:.2f}\n\n"

    # ===== VALUATION ANALYSIS =====
    result += "## 3. Valuation Analysis (Fair Value Calculation)\n\n"
    result += "This section calculates fair value using multiple methodologies, showing the complete logic and calculations for each approach.\n\n"

    fair_values = {}

    # Calculate shares outstanding from market cap
    mkt_cap = profile.get("mktCap", 0)
    shares_outstanding = mkt_cap / current_price if current_price and mkt_cap else 0

    # Method 1: FMP DCF Value
    if dcf_data:
        dcf_value = dcf_data.get("dcf", 0)
        stock_price = dcf_data.get("Stock Price", current_price)
        fair_values["DCF (FMP)"] = dcf_value

        result += "### 3.1 Discounted Cash Flow (DCF) Valuation\n\n"
        result += "**Methodology:** DCF values a company based on the present value of its expected future cash flows.\n\n"
        result += "**Formula:**\n"
        result += "```\n"
        result += "Intrinsic Value = Σ (FCFt / (1 + WACC)^t) + Terminal Value / (1 + WACC)^n\n"
        result += "```\n\n"
        result += "**FMP Calculated DCF:**\n"
        result += f"- DCF Fair Value: **${dcf_value:.2f}**\n"
        result += f"- Current Market Price: ${current_price:.2f}\n"
        upside = ((dcf_value / current_price) - 1) * 100 if current_price else 0
        result += f"- Implied Upside/Downside: **{upside:+.1f}%**\n\n"

        result += "**Interpretation:**\n"
        if upside > 20:
            result += f"- The DCF model suggests the stock is **significantly undervalued** by {upside:.0f}%\n"
            result += "- The market may be underestimating future cash flow potential\n"
        elif upside > 0:
            result += f"- The DCF model suggests the stock is **modestly undervalued** by {upside:.0f}%\n"
        elif upside > -20:
            result += f"- The DCF model suggests the stock is **fairly valued** or slightly overvalued\n"
        else:
            result += f"- The DCF model suggests the stock is **overvalued** by {abs(upside):.0f}%\n"
        result += "\n"

    # Method 2: Graham Number (Detailed)
    if balance_sheet and income_stmt:
        latest_bs = balance_sheet[0]
        latest_income = income_stmt[0]
        eps = latest_income.get("eps", 0) or 0
        total_equity = latest_bs.get("totalStockholdersEquity", 0)

        # Calculate book value per share properly
        if shares_outstanding > 0:
            book_value_per_share = total_equity / shares_outstanding
        else:
            book_value_per_share = 0

        result += "### 3.2 Graham Number (Value Investing)\n\n"
        result += "**Methodology:** Developed by Benjamin Graham, the father of value investing. It calculates the maximum price a defensive investor should pay for a stock.\n\n"
        result += "**Formula:**\n"
        result += "```\n"
        result += "Graham Number = √(22.5 × EPS × Book Value Per Share)\n"
        result += "```\n\n"
        result += "**Where:**\n"
        result += "- 22.5 = Graham's constant (derived from P/E of 15 × P/B of 1.5)\n"
        result += "- EPS = Earnings Per Share (trailing twelve months)\n"
        result += "- BVPS = Total Shareholders' Equity / Shares Outstanding\n\n"

        result += "**Step-by-Step Calculation:**\n"
        result += f"1. **EPS (Earnings Per Share):** ${eps:.2f}\n"
        result += f"   - Net Income: ${latest_income.get('netIncome', 0):,.0f}\n"
        result += f"   - Shares Outstanding: {shares_outstanding:,.0f}\n\n"

        result += f"2. **Book Value Per Share:** ${book_value_per_share:.2f}\n"
        result += f"   - Total Shareholders' Equity: ${total_equity:,.0f}\n"
        result += f"   - Shares Outstanding: {shares_outstanding:,.0f}\n"
        result += f"   - BVPS = ${total_equity:,.0f} / {shares_outstanding:,.0f} = ${book_value_per_share:.2f}\n\n"

        if eps > 0 and book_value_per_share > 0:
            graham_number = (22.5 * eps * book_value_per_share) ** 0.5
            fair_values["Graham Number"] = graham_number

            result += f"3. **Graham Number Calculation:**\n"
            result += f"   - Graham Number = √(22.5 × ${eps:.2f} × ${book_value_per_share:.2f})\n"
            result += f"   - Graham Number = √({22.5 * eps * book_value_per_share:,.2f})\n"
            result += f"   - **Graham Number = ${graham_number:.2f}**\n\n"

            upside = ((graham_number / current_price) - 1) * 100 if current_price else 0
            result += f"**Result:**\n"
            result += f"- Graham Fair Value: **${graham_number:.2f}**\n"
            result += f"- Current Price: ${current_price:.2f}\n"
            result += f"- Implied Upside/Downside: **{upside:+.1f}%**\n\n"

            result += "**Interpretation:**\n"
            if current_price < graham_number:
                result += f"- Current price is **below** the Graham Number, suggesting a margin of safety\n"
                result += f"- Graham would consider this potentially attractive for value investors\n"
            else:
                result += f"- Current price is **above** the Graham Number\n"
                result += f"- Graham would suggest waiting for a lower entry point\n"
            result += "\n"
        else:
            result += "- *Cannot calculate Graham Number: EPS or Book Value is negative/zero*\n\n"

    # Method 3: Relative Valuation (Peer Comparison) - Detailed
    if peer_quotes and not ratios_df.empty:
        result += "### 3.3 Relative Valuation (Peer P/E Comparison)\n\n"
        result += "**Methodology:** Compare the company's valuation multiples to sector peers to determine if it's trading at a premium or discount.\n\n"
        result += "**Formula:**\n"
        result += "```\n"
        result += "Fair Value = Company EPS × Sector Average P/E\n"
        result += "```\n\n"

        peer_data = [(q.symbol, q.pe, q.price) for q in peer_quotes if q.pe and q.pe > 0 and q.pe < 100]

        if peer_data:
            result += "**Peer P/E Ratios:**\n"
            result += "| Peer | P/E Ratio | Price |\n"
            result += "|------|-----------|-------|\n"
            for sym, pe, price in sorted(peer_data, key=lambda x: x[1])[:10]:
                result += f"| {sym} | {pe:.1f}x | ${price:.2f} |\n"
            result += "\n"

            peer_pes = [p[1] for p in peer_data]
            avg_peer_pe = sum(peer_pes) / len(peer_pes)
            median_peer_pe = sorted(peer_pes)[len(peer_pes)//2]
            min_peer_pe = min(peer_pes)
            max_peer_pe = max(peer_pes)

            eps = income_stmt[0].get("eps", 0) if income_stmt else 0
            company_pe = current_price / eps if eps > 0 else 0

            result += "**Peer Statistics:**\n"
            result += f"- Minimum P/E: {min_peer_pe:.1f}x\n"
            result += f"- Average P/E: {avg_peer_pe:.1f}x\n"
            result += f"- Median P/E: {median_peer_pe:.1f}x\n"
            result += f"- Maximum P/E: {max_peer_pe:.1f}x\n\n"

            result += f"**Company's Current P/E:** {company_pe:.1f}x\n\n"

            if eps > 0:
                fair_value_avg_pe = eps * avg_peer_pe
                fair_value_median_pe = eps * median_peer_pe
                fair_values["Peer Avg P/E"] = fair_value_avg_pe
                fair_values["Peer Median P/E"] = fair_value_median_pe

                result += "**Step-by-Step Calculation:**\n"
                result += f"1. Company EPS: ${eps:.2f}\n"
                result += f"2. Sector Average P/E: {avg_peer_pe:.1f}x\n"
                result += f"3. Sector Median P/E: {median_peer_pe:.1f}x\n\n"

                result += f"**Fair Value (Average P/E):**\n"
                result += f"- Fair Value = ${eps:.2f} × {avg_peer_pe:.1f} = **${fair_value_avg_pe:.2f}**\n"
                upside_avg = ((fair_value_avg_pe / current_price) - 1) * 100 if current_price else 0
                result += f"- Implied Upside/Downside: **{upside_avg:+.1f}%**\n\n"

                result += f"**Fair Value (Median P/E):**\n"
                result += f"- Fair Value = ${eps:.2f} × {median_peer_pe:.1f} = **${fair_value_median_pe:.2f}**\n"
                upside_median = ((fair_value_median_pe / current_price) - 1) * 100 if current_price else 0
                result += f"- Implied Upside/Downside: **{upside_median:+.1f}%**\n\n"

                result += "**Interpretation:**\n"
                if company_pe < avg_peer_pe:
                    discount = ((avg_peer_pe - company_pe) / avg_peer_pe) * 100
                    result += f"- Stock is trading at a **{discount:.0f}% discount** to sector average\n"
                    result += "- May indicate undervaluation or company-specific concerns\n"
                else:
                    premium = ((company_pe - avg_peer_pe) / avg_peer_pe) * 100
                    result += f"- Stock is trading at a **{premium:.0f}% premium** to sector average\n"
                    result += "- Premium may be justified by superior growth or quality\n"
                result += "\n"

    # Method 4: Custom DCF Model (Detailed)
    if cash_flow and balance_sheet:
        fcf = cash_flow[0].get("freeCashFlow", 0)

        if fcf > 0 and shares_outstanding > 0:
            result += "### 3.4 Custom DCF Model (5-Year Projection)\n\n"
            result += "**Methodology:** Project future free cash flows, discount to present value, add terminal value.\n\n"
            result += "**Formula:**\n"
            result += "```\n"
            result += "Enterprise Value = Σ (FCF × (1+g)^t / (1+r)^t) + Terminal Value\n"
            result += "Equity Value = Enterprise Value - Net Debt + Cash\n"
            result += "Fair Value Per Share = Equity Value / Shares Outstanding\n"
            result += "```\n\n"

            # Assumptions
            growth_rate = 0.10  # 10% growth
            terminal_growth = 0.025  # 2.5% terminal
            discount_rate = 0.10  # 10% WACC

            result += "**Assumptions:**\n"
            result += f"- Latest Free Cash Flow (FCF): ${fcf:,.0f}\n"
            result += f"- FCF Growth Rate (Years 1-5): {growth_rate*100:.0f}% annually\n"
            result += f"- Terminal Growth Rate: {terminal_growth*100:.1f}% (perpetuity)\n"
            result += f"- Discount Rate (WACC): {discount_rate*100:.0f}%\n\n"

            # Calculate projected FCFs
            result += "**Step 1: Project Future Cash Flows**\n"
            result += "| Year | FCF | Discount Factor | Present Value |\n"
            result += "|------|-----|-----------------|---------------|\n"

            projected_fcfs = []
            pv_fcfs = []
            for year in range(1, 6):
                projected_fcf = fcf * ((1 + growth_rate) ** year)
                discount_factor = 1 / ((1 + discount_rate) ** year)
                pv = projected_fcf * discount_factor
                projected_fcfs.append(projected_fcf)
                pv_fcfs.append(pv)
                result += f"| {year} | ${projected_fcf:,.0f} | {discount_factor:.4f} | ${pv:,.0f} |\n"

            sum_pv_fcf = sum(pv_fcfs)
            result += f"| **Total** | | | **${sum_pv_fcf:,.0f}** |\n\n"

            # Terminal Value
            result += "**Step 2: Calculate Terminal Value**\n"
            terminal_fcf = projected_fcfs[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            discount_factor_5 = 1 / ((1 + discount_rate) ** 5)
            pv_terminal = terminal_value * discount_factor_5

            result += f"- Year 5 FCF: ${projected_fcfs[-1]:,.0f}\n"
            result += f"- Terminal FCF (Year 6): ${projected_fcfs[-1]:,.0f} × (1 + {terminal_growth*100:.1f}%) = ${terminal_fcf:,.0f}\n"
            result += f"- Terminal Value = ${terminal_fcf:,.0f} / ({discount_rate*100:.0f}% - {terminal_growth*100:.1f}%) = ${terminal_value:,.0f}\n"
            result += f"- PV of Terminal Value = ${terminal_value:,.0f} × {discount_factor_5:.4f} = ${pv_terminal:,.0f}\n\n"

            # Enterprise Value
            result += "**Step 3: Calculate Enterprise Value**\n"
            enterprise_value = sum_pv_fcf + pv_terminal
            result += f"- PV of FCFs (Years 1-5): ${sum_pv_fcf:,.0f}\n"
            result += f"- PV of Terminal Value: ${pv_terminal:,.0f}\n"
            result += f"- **Enterprise Value: ${enterprise_value:,.0f}**\n\n"

            # Equity Value
            result += "**Step 4: Calculate Equity Value**\n"
            total_debt = balance_sheet[0].get("totalDebt", 0) if balance_sheet else 0
            cash = balance_sheet[0].get("cashAndCashEquivalents", 0) if balance_sheet else 0
            equity_value = enterprise_value - total_debt + cash

            result += f"- Enterprise Value: ${enterprise_value:,.0f}\n"
            result += f"- Less: Total Debt: (${total_debt:,.0f})\n"
            result += f"- Plus: Cash & Equivalents: ${cash:,.0f}\n"
            result += f"- **Equity Value: ${equity_value:,.0f}**\n\n"

            # Fair Value Per Share
            result += "**Step 5: Calculate Fair Value Per Share**\n"
            fair_value_dcf = equity_value / shares_outstanding
            fair_values["Custom DCF"] = fair_value_dcf

            result += f"- Equity Value: ${equity_value:,.0f}\n"
            result += f"- Shares Outstanding: {shares_outstanding:,.0f}\n"
            result += f"- **Fair Value Per Share: ${equity_value:,.0f} / {shares_outstanding:,.0f} = ${fair_value_dcf:.2f}**\n\n"

            upside = ((fair_value_dcf / current_price) - 1) * 100 if current_price else 0
            result += f"**Result:**\n"
            result += f"- DCF Fair Value: **${fair_value_dcf:.2f}**\n"
            result += f"- Current Price: ${current_price:.2f}\n"
            result += f"- Implied Upside/Downside: **{upside:+.1f}%**\n\n"

            result += "**Sensitivity Note:** DCF is sensitive to assumptions. A 1% change in:\n"
            result += "- Growth rate changes fair value by ~5-10%\n"
            result += "- Discount rate changes fair value by ~10-15%\n\n"

    # Method 5: Earnings Power Value (EPV)
    if income_stmt and not ratios_df.empty:
        result += "### 3.5 Earnings Power Value (EPV)\n\n"
        result += "**Methodology:** Values the company based on its current normalized earnings, assuming no growth.\n\n"
        result += "**Formula:**\n"
        result += "```\n"
        result += "EPV = Adjusted Earnings / Cost of Capital\n"
        result += "```\n\n"

        operating_income = income_stmt[0].get("operatingIncome", 0)
        tax_rate = 0.25  # Assume 25% tax rate
        adjusted_earnings = operating_income * (1 - tax_rate)
        cost_of_capital = 0.10  # 10% WACC

        result += "**Calculation:**\n"
        result += f"1. Operating Income (EBIT): ${operating_income:,.0f}\n"
        result += f"2. Tax Rate (assumed): {tax_rate*100:.0f}%\n"
        result += f"3. After-Tax Earnings: ${operating_income:,.0f} × (1 - {tax_rate*100:.0f}%) = ${adjusted_earnings:,.0f}\n"
        result += f"4. Cost of Capital (WACC): {cost_of_capital*100:.0f}%\n\n"

        if adjusted_earnings > 0:
            epv_enterprise = adjusted_earnings / cost_of_capital
            total_debt = balance_sheet[0].get("totalDebt", 0) if balance_sheet else 0
            cash = balance_sheet[0].get("cashAndCashEquivalents", 0) if balance_sheet else 0
            epv_equity = epv_enterprise - total_debt + cash

            result += f"5. EPV (Enterprise): ${adjusted_earnings:,.0f} / {cost_of_capital*100:.0f}% = ${epv_enterprise:,.0f}\n"
            result += f"6. Less Debt, Plus Cash: ${epv_enterprise:,.0f} - ${total_debt:,.0f} + ${cash:,.0f} = ${epv_equity:,.0f}\n"

            if shares_outstanding > 0:
                epv_per_share = epv_equity / shares_outstanding
                fair_values["EPV"] = epv_per_share
                upside = ((epv_per_share / current_price) - 1) * 100 if current_price else 0

                result += f"7. **EPV Per Share: ${epv_equity:,.0f} / {shares_outstanding:,.0f} = ${epv_per_share:.2f}**\n\n"
                result += f"**Result:** EPV Fair Value = **${epv_per_share:.2f}** ({upside:+.1f}% vs current)\n\n"

    # Fair Value Summary
    result += "### 3.6 Fair Value Summary\n\n"
    result += "| Method | Fair Value | vs Current Price | Assessment |\n"
    result += "|--------|------------|------------------|------------|\n"

    for method, fv in fair_values.items():
        diff = ((fv / current_price) - 1) * 100 if current_price else 0
        if diff > 20:
            status = "🟢 Significantly Undervalued"
        elif diff > 5:
            status = "🟢 Undervalued"
        elif diff > -5:
            status = "🟡 Fairly Valued"
        elif diff > -20:
            status = "🟠 Overvalued"
        else:
            status = "🔴 Significantly Overvalued"
        result += f"| {method} | ${fv:.2f} | {diff:+.1f}% | {status} |\n"

    if fair_values:
        avg_fair_value = sum(fair_values.values()) / len(fair_values)
        min_fair_value = min(fair_values.values())
        max_fair_value = max(fair_values.values())
        avg_diff = ((avg_fair_value / current_price) - 1) * 100 if current_price else 0

        result += f"\n**Valuation Range:**\n"
        result += f"- Low Estimate: ${min_fair_value:.2f}\n"
        result += f"- Average Estimate: **${avg_fair_value:.2f}**\n"
        result += f"- High Estimate: ${max_fair_value:.2f}\n"
        result += f"- Current Price: ${current_price:.2f}\n"
        result += f"- Average Implied Upside/Downside: **{avg_diff:+.1f}%**\n\n"

        result += "**Valuation Conclusion:**\n"
        methods_above = sum(1 for fv in fair_values.values() if fv > current_price)
        methods_below = len(fair_values) - methods_above

        if methods_above > methods_below:
            result += f"- {methods_above} of {len(fair_values)} valuation methods suggest the stock is **undervalued**\n"
        elif methods_below > methods_above:
            result += f"- {methods_below} of {len(fair_values)} valuation methods suggest the stock is **overvalued**\n"
        else:
            result += f"- Valuation methods are split, suggesting stock is **fairly valued**\n"
        result += "\n"

    # ===== QUANTITATIVE ANALYSIS =====
    result += "## 4. Quantitative Analysis\n\n"

    if not hist.empty:
        prices = hist["adjClose"].values

        # Technical Indicators
        result += "### 4.1 Technical Indicators\n\n"

        # Moving Averages
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
        sma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else prices[-1]

        result += "**Moving Averages:**\n"
        result += f"- 20-day SMA: ${sma_20:.2f} ({((current_price/sma_20)-1)*100:+.1f}%)\n"
        result += f"- 50-day SMA: ${sma_50:.2f} ({((current_price/sma_50)-1)*100:+.1f}%)\n"
        result += f"- 200-day SMA: ${sma_200:.2f} ({((current_price/sma_200)-1)*100:+.1f}%)\n"

        # Trend assessment
        if current_price > sma_20 > sma_50 > sma_200:
            result += "- Trend: 📈 **STRONG UPTREND** (Price > 20 > 50 > 200 SMA)\n"
        elif current_price < sma_20 < sma_50 < sma_200:
            result += "- Trend: 📉 **STRONG DOWNTREND** (Price < 20 < 50 < 200 SMA)\n"
        elif current_price > sma_200:
            result += "- Trend: ⬆️ **BULLISH** (Above 200 SMA)\n"
        else:
            result += "- Trend: ⬇️ **BEARISH** (Below 200 SMA)\n"

        # RSI
        changes = np.diff(prices)
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi = 100 - (100 / (1 + rs))

        result += f"\n**RSI (14): {rsi:.1f}**\n"
        if rsi > 70:
            result += "- Status: 🔴 OVERBOUGHT\n"
        elif rsi < 30:
            result += "- Status: 🟢 OVERSOLD\n"
        else:
            result += "- Status: ⚪ NEUTRAL\n"

        # Volatility
        returns = np.diff(prices) / prices[:-1]
        volatility_daily = np.std(returns)
        volatility_annual = volatility_daily * np.sqrt(252) * 100

        result += f"\n**Volatility:**\n"
        result += f"- Daily: {volatility_daily*100:.2f}%\n"
        result += f"- Annualized: {volatility_annual:.1f}%\n"

        # Risk Metrics
        result += "\n### 4.2 Risk Metrics\n\n"

        # VaR
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        result += f"- 95% VaR (Daily): {var_95:.2f}%\n"
        result += f"- 99% VaR (Daily): {var_99:.2f}%\n"

        # Max Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100

        result += f"- Max Drawdown (1Y): {max_drawdown:.1f}%\n"
        result += f"- Beta: {profile.get('beta', 1.0):.2f}\n"

        # Sharpe approximation
        avg_return = np.mean(returns) * 252
        sharpe = (avg_return - 0.05) / (volatility_annual / 100) if volatility_annual > 0 else 0
        result += f"- Sharpe Ratio (Est): {sharpe:.2f}\n\n"

    # ===== ANALYST SENTIMENT =====
    result += "## 5. Analyst Sentiment\n\n"

    if estimates:
        result += "### 5.1 Earnings Estimates\n\n"
        for est in estimates[:2]:
            result += f"**{est.get('date', 'N/A')}:**\n"
            result += f"- Revenue Est: ${est.get('estimatedRevenueAvg', 0):,.0f}\n"
            result += f"- EPS Est: ${est.get('estimatedEpsAvg', 0):.2f}\n"

    if earnings:
        latest_earnings = earnings[0]
        actual = latest_earnings.get("actualEarningResult", 0)
        estimated = latest_earnings.get("estimatedEarning", 0)
        if estimated:
            surprise = ((actual - estimated) / abs(estimated)) * 100
            result += f"\n### 5.2 Latest Earnings\n"
            result += f"- Actual EPS: ${actual:.2f}\n"
            result += f"- Estimated EPS: ${estimated:.2f}\n"
            result += f"- Surprise: {surprise:+.1f}%\n"

    if recommendations:
        result += "\n### 5.3 Analyst Recommendations\n\n"
        for rec in recommendations[:3]:
            result += f"- {rec.get('date', 'N/A')}: {rec.get('analystRatingsStrongBuy', 0)} Strong Buy, "
            result += f"{rec.get('analystRatingsBuy', 0)} Buy, {rec.get('analystRatingsHold', 0)} Hold, "
            result += f"{rec.get('analystRatingsSell', 0)} Sell\n"

    # ===== FINAL RECOMMENDATION =====
    result += "\n## 6. Investment Recommendation\n\n"

    # Score calculation
    scores = []

    # Valuation score
    if fair_values:
        avg_fair_value = sum(fair_values.values()) / len(fair_values)
        valuation_score = min(100, max(0, 50 + (avg_fair_value / current_price - 1) * 100)) if current_price else 50
        scores.append(("Valuation", valuation_score))

    # Profitability score
    if not ratios_df.empty:
        r = ratios_df.iloc[0]
        roe = r.get('returnOnEquity', 0) * 100
        margin = r.get('netProfitMargin', 0) * 100
        prof_score = min(100, (roe * 2 + margin * 2))
        scores.append(("Profitability", prof_score))

    # Technical score
    if not hist.empty:
        tech_score = 50
        if current_price > sma_50:
            tech_score += 15
        if current_price > sma_200:
            tech_score += 15
        if 30 < rsi < 70:
            tech_score += 10
        elif rsi < 30:
            tech_score += 20  # Oversold opportunity
        scores.append(("Technical", min(100, tech_score)))

    # Risk score (inverse - lower risk = higher score)
    if not hist.empty:
        risk_score = max(0, 100 - volatility_annual)
        scores.append(("Risk-Adjusted", risk_score))

    if scores:
        overall_score = sum(s[1] for s in scores) / len(scores)

        result += "**Scoring Summary:**\n"
        for name, score in scores:
            bar = "█" * int(score/10) + "░" * (10 - int(score/10))
            result += f"- {name}: {bar} {score:.0f}/100\n"

        result += f"\n**Overall Score: {overall_score:.0f}/100**\n\n"

        # Recommendation
        result += "**RECOMMENDATION:** "
        if overall_score >= 70 and avg_fair_value > current_price * 1.1:
            result += "🟢 **STRONG BUY**\n"
            result += "- Fundamentals are strong\n"
            result += "- Stock appears undervalued\n"
            result += "- Technical setup is favorable\n"
        elif overall_score >= 60 and avg_fair_value > current_price:
            result += "🟢 **BUY**\n"
            result += "- Decent fundamentals\n"
            result += "- Trading below fair value\n"
        elif overall_score >= 50 or (avg_fair_value > current_price * 0.9 and avg_fair_value < current_price * 1.1):
            result += "🟡 **HOLD**\n"
            result += "- Trading near fair value\n"
            result += "- Wait for better entry or exit point\n"
        elif overall_score >= 40:
            result += "🟠 **REDUCE**\n"
            result += "- Some concerns with valuation or fundamentals\n"
            result += "- Consider taking partial profits\n"
        else:
            result += "🔴 **SELL**\n"
            result += "- Stock appears overvalued\n"
            result += "- Fundamental or technical concerns\n"

    result += "\n---\n*Disclaimer: This analysis is for informational purposes only and does not constitute investment advice.*\n"

    return result


@tool
def comprehensive_sector_analysis_tool(
    sector: str,
) -> str:
    """Perform complete sector analysis comparing all equities fundamentally and quantitatively.

    This tool analyzes an entire sector by:
    - Gathering fundamental data for all major companies in the sector
    - Calculating fair values for each company using multiple methods
    - Comparing valuations across the sector
    - Identifying the most undervalued and overvalued stocks
    - Providing sector-level insights and recommendations

    Use this tool when you need a complete picture of a sector and its best investment opportunities.

    Args:
        sector: Sector name (e.g., "Technology", "Healthcare", "Financial Services",
                "Consumer Cyclical", "Industrials", "Energy", "Basic Materials",
                "Communication Services", "Consumer Defensive", "Utilities", "Real Estate")

    Returns:
        String with comprehensive sector analysis including best opportunities.
    """
    data_pipeline, _ = get_pipeline()

    async def fetch():
        # Get all stocks in sector
        stocks = await data_pipeline._stock_list.get_stock_screener(
            sector=sector,
            is_actively_trading=True,
            limit=50,
        )

        if not stocks:
            return None, None, None, None, None

        # Get top stocks by market cap
        symbols = [s["symbol"] for s in stocks[:30]]

        # Get quotes
        quotes = await data_pipeline.get_quotes_batch(symbols)

        # Get profiles
        profiles = await data_pipeline.get_company_profiles_batch(symbols[:25])

        # Get ratios for top companies
        ratios_data = {}
        for sym in symbols[:15]:
            try:
                ratios = await data_pipeline.get_financial_ratios(sym, "annual", limit=1)
                if not ratios.empty:
                    ratios_data[sym] = ratios.iloc[0].to_dict()
            except Exception:
                pass

        # Get DCF values
        dcf_data = {}
        for sym in symbols[:15]:
            try:
                dcf = await data_pipeline._company_info.get_dcf(sym)
                if dcf:
                    dcf_data[sym] = dcf
            except Exception:
                pass

        # Get sector performance
        sector_perf = await data_pipeline.get_sector_performance()

        await data_pipeline._client.close()
        return quotes, profiles, ratios_data, dcf_data, sector_perf

    quotes, profiles, ratios_data, dcf_data, sector_perf = run_async(fetch())

    if not quotes:
        return f"No data found for sector: {sector}"

    import numpy as np

    result = f"# Comprehensive Sector Analysis: {sector}\n\n"

    # ===== SECTOR OVERVIEW =====
    result += "## 1. Sector Overview\n\n"

    # Sector performance
    sector_change = None
    for _, row in sector_perf.iterrows():
        if sector.lower() in row.get("sector", "").lower():
            sector_change = float(row.get("changesPercentage", "0").replace("%", ""))
            break

    if sector_change is not None:
        emoji = "📈" if sector_change >= 0 else "📉"
        result += f"**Sector Performance Today:** {emoji} {sector_change:+.2f}%\n\n"

    # Summary statistics
    total_mkt_cap = sum(q.market_cap or 0 for q in quotes)
    avg_pe = np.mean([q.pe for q in quotes if q.pe and 0 < q.pe < 100])
    avg_change = np.mean([q.changes_percentage or 0 for q in quotes])

    result += f"**Sector Statistics:**\n"
    result += f"- Total Market Cap: ${total_mkt_cap:,.0f}\n"
    result += f"- Average P/E: {avg_pe:.1f}x\n"
    result += f"- Average Stock Change: {avg_change:+.2f}%\n"
    result += f"- Companies Analyzed: {len(quotes)}\n\n"

    # ===== COMPANY COMPARISON =====
    result += "## 2. Company Comparison\n\n"

    # Build comparison table
    companies = []
    profile_lookup = {p.symbol: p for p in profiles}

    for q in quotes:
        company = {
            "symbol": q.symbol,
            "name": q.name,
            "price": q.price,
            "change": q.changes_percentage or 0,
            "mkt_cap": q.market_cap or 0,
            "pe": q.pe if q.pe and 0 < q.pe < 200 else None,
            "volume_ratio": (q.volume / q.avg_volume) if q.avg_volume else 1,
        }

        # Add profile data
        profile = profile_lookup.get(q.symbol)
        if profile:
            company["industry"] = profile.industry
            company["beta"] = profile.beta

        # Add ratios
        ratios = ratios_data.get(q.symbol, {})
        company["roe"] = ratios.get("returnOnEquity", 0)
        company["margin"] = ratios.get("netProfitMargin", 0)
        company["debt_equity"] = ratios.get("debtEquityRatio", 0)
        company["pb"] = ratios.get("priceToBookRatio", 0)
        company["ps"] = ratios.get("priceToSalesRatio", 0)

        # Add DCF data
        dcf = dcf_data.get(q.symbol, {})
        company["dcf_value"] = dcf.get("dcf", 0) if dcf else 0
        if company["dcf_value"] and company["price"]:
            company["dcf_upside"] = ((company["dcf_value"] / company["price"]) - 1) * 100
        else:
            company["dcf_upside"] = 0

        companies.append(company)

    # Sort by market cap
    companies.sort(key=lambda x: -x["mkt_cap"])

    # Top Companies Table
    result += "### 2.1 Top Companies by Market Cap\n\n"
    result += "| Company | Price | Change | P/E | ROE | Margin | DCF Value | Upside |\n"
    result += "|---------|-------|--------|-----|-----|--------|-----------|--------|\n"

    for c in companies[:15]:
        pe_str = f"{c['pe']:.1f}x" if c['pe'] else "N/A"
        roe_str = f"{c['roe']*100:.1f}%" if c['roe'] else "N/A"
        margin_str = f"{c['margin']*100:.1f}%" if c['margin'] else "N/A"
        dcf_str = f"${c['dcf_value']:.0f}" if c['dcf_value'] else "N/A"
        upside_str = f"{c['dcf_upside']:+.0f}%" if c['dcf_upside'] else "N/A"

        result += f"| **{c['symbol']}** | ${c['price']:.2f} | {c['change']:+.1f}% | {pe_str} | {roe_str} | {margin_str} | {dcf_str} | {upside_str} |\n"

    # ===== VALUATION ANALYSIS =====
    result += "\n## 3. Valuation Analysis\n\n"

    # Find undervalued stocks (DCF upside > 20%)
    undervalued = [c for c in companies if c["dcf_upside"] > 20]
    undervalued.sort(key=lambda x: -x["dcf_upside"])

    if undervalued:
        result += "### 3.1 Most Undervalued (DCF Upside > 20%)\n\n"
        for c in undervalued[:7]:
            result += f"**{c['symbol']}** - {c['name']}\n"
            result += f"- Current Price: ${c['price']:.2f}\n"
            result += f"- DCF Fair Value: ${c['dcf_value']:.2f}\n"
            result += f"- **Upside Potential: {c['dcf_upside']:+.1f}%**\n"
            if c['pe']:
                pe_vs_sector = ((c['pe'] / avg_pe) - 1) * 100
                result += f"- P/E: {c['pe']:.1f}x ({pe_vs_sector:+.1f}% vs sector avg)\n"
            result += "\n"

    # Find overvalued stocks (DCF upside < -20%)
    overvalued = [c for c in companies if c["dcf_upside"] < -20]
    overvalued.sort(key=lambda x: x["dcf_upside"])

    if overvalued:
        result += "### 3.2 Most Overvalued (DCF Downside > 20%)\n\n"
        for c in overvalued[:5]:
            result += f"**{c['symbol']}** - {c['name']}\n"
            result += f"- Current Price: ${c['price']:.2f}\n"
            result += f"- DCF Fair Value: ${c['dcf_value']:.2f}\n"
            result += f"- **Downside Risk: {c['dcf_upside']:.1f}%**\n\n"

    # ===== FUNDAMENTAL SCREENING =====
    result += "## 4. Fundamental Screening\n\n"

    # High Quality (High ROE, Good Margins)
    quality = [c for c in companies if c['roe'] and c['roe'] > 0.15 and c['margin'] and c['margin'] > 0.10]
    quality.sort(key=lambda x: -x['roe'])

    if quality:
        result += "### 4.1 Quality Leaders (ROE > 15%, Margin > 10%)\n\n"
        for c in quality[:5]:
            result += f"- **{c['symbol']}**: ROE {c['roe']*100:.1f}%, Margin {c['margin']*100:.1f}%\n"
        result += "\n"

    # Value Opportunities (Low P/E, Positive ROE)
    value = [c for c in companies if c['pe'] and c['pe'] < avg_pe * 0.7 and c['roe'] and c['roe'] > 0.05]
    value.sort(key=lambda x: x['pe'])

    if value:
        result += "### 4.2 Value Opportunities (Low P/E, Profitable)\n\n"
        for c in value[:5]:
            result += f"- **{c['symbol']}**: P/E {c['pe']:.1f}x (vs sector {avg_pe:.1f}x), ROE {c['roe']*100:.1f}%\n"
        result += "\n"

    # Momentum (Top Performers Today)
    momentum = sorted(companies, key=lambda x: -x['change'])[:5]

    result += "### 4.3 Momentum Leaders (Top Performers Today)\n\n"
    for c in momentum:
        vol_str = f"({c['volume_ratio']:.1f}x vol)" if c['volume_ratio'] > 1.5 else ""
        result += f"- **{c['symbol']}**: {c['change']:+.2f}% {vol_str}\n"
    result += "\n"

    # Laggards
    laggards = sorted(companies, key=lambda x: x['change'])[:5]

    result += "### 4.4 Today's Laggards (Potential Opportunities?)\n\n"
    for c in laggards:
        result += f"- **{c['symbol']}**: {c['change']:+.2f}%"
        if c['dcf_upside'] > 10:
            result += " *(Still undervalued per DCF)*"
        result += "\n"

    # ===== SECTOR METRICS =====
    result += "\n## 5. Sector Metrics Distribution\n\n"

    # P/E distribution
    pe_values = [c['pe'] for c in companies if c['pe']]
    if pe_values:
        result += "**P/E Ratio:**\n"
        result += f"- Low: {min(pe_values):.1f}x\n"
        result += f"- Median: {sorted(pe_values)[len(pe_values)//2]:.1f}x\n"
        result += f"- High: {max(pe_values):.1f}x\n"
        result += f"- Average: {np.mean(pe_values):.1f}x\n\n"

    # ROE distribution
    roe_values = [c['roe']*100 for c in companies if c['roe']]
    if roe_values:
        result += "**ROE (Return on Equity):**\n"
        result += f"- Low: {min(roe_values):.1f}%\n"
        result += f"- Median: {sorted(roe_values)[len(roe_values)//2]:.1f}%\n"
        result += f"- High: {max(roe_values):.1f}%\n"
        result += f"- Average: {np.mean(roe_values):.1f}%\n\n"

    # ===== TOP PICKS =====
    result += "## 6. Sector Top Picks\n\n"

    # Score each company
    scored_companies = []
    for c in companies:
        score = 0

        # Valuation score (DCF upside)
        if c['dcf_upside'] > 30:
            score += 30
        elif c['dcf_upside'] > 15:
            score += 20
        elif c['dcf_upside'] > 0:
            score += 10

        # Quality score (ROE)
        if c['roe'] and c['roe'] > 0.20:
            score += 25
        elif c['roe'] and c['roe'] > 0.15:
            score += 20
        elif c['roe'] and c['roe'] > 0.10:
            score += 15
        elif c['roe'] and c['roe'] > 0.05:
            score += 10

        # Margin score
        if c['margin'] and c['margin'] > 0.20:
            score += 20
        elif c['margin'] and c['margin'] > 0.10:
            score += 15
        elif c['margin'] and c['margin'] > 0.05:
            score += 10

        # Value score (P/E below sector)
        if c['pe'] and avg_pe:
            if c['pe'] < avg_pe * 0.7:
                score += 15
            elif c['pe'] < avg_pe * 0.9:
                score += 10

        # Low debt score
        if c['debt_equity'] and c['debt_equity'] < 0.5:
            score += 10

        c['score'] = score
        if score > 0:
            scored_companies.append(c)

    scored_companies.sort(key=lambda x: -x['score'])

    result += "### Best Investment Opportunities (Composite Score)\n\n"
    for i, c in enumerate(scored_companies[:10], 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        result += f"**{medal} {c['symbol']}** - Score: {c['score']}/100\n"
        result += f"   - Price: ${c['price']:.2f} | "
        result += f"DCF Upside: {c['dcf_upside']:+.0f}% | " if c['dcf_upside'] else ""
        result += f"P/E: {c['pe']:.1f}x | " if c['pe'] else ""
        result += f"ROE: {c['roe']*100:.0f}%\n" if c['roe'] else "\n"

    result += "\n---\n*Disclaimer: This analysis is for informational purposes only and does not constitute investment advice.*\n"

    return result


@tool
def correlation_analysis_tool(
    symbols: str,
    days: int = 60,
) -> str:
    """Analyze correlations between multiple securities.

    Use this tool to understand how different stocks, sectors, or asset classes
    move together. Helps identify diversification opportunities and hidden relationships.

    Args:
        symbols: Comma-separated symbols (e.g., "AAPL,MSFT,GOOGL,SPY,QQQ")
        days: Number of days for correlation calculation (default: 60)

    Returns:
        String with correlation matrix and analysis.
    """
    data_pipeline, _ = get_pipeline()

    async def fetch():
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        to_date = date.today()
        from_date = to_date - timedelta(days=days + 10)

        returns_df = await data_pipeline.get_returns_matrix(symbol_list, str(from_date), str(to_date))
        await data_pipeline._client.close()
        return returns_df, symbol_list

    returns_df, symbol_list = run_async(fetch())

    if returns_df.empty:
        return "Unable to fetch data for the specified symbols."

    import numpy as np

    result = f"**Correlation Analysis ({days} Days)**\n\n"
    result += f"Symbols: {', '.join(symbol_list)}\n\n"

    # Calculate correlation matrix
    corr_matrix = returns_df.corr()

    result += "**Correlation Matrix:**\n```\n"

    # Header
    result += " " * 8
    for sym in corr_matrix.columns:
        result += f"{sym:>8}"
    result += "\n"

    # Matrix
    for sym in corr_matrix.index:
        result += f"{sym:<8}"
        for col in corr_matrix.columns:
            val = corr_matrix.loc[sym, col]
            result += f"{val:>8.2f}"
        result += "\n"

    result += "```\n\n"

    # Find notable correlations
    high_corr = []
    low_corr = []
    negative_corr = []

    for i, sym1 in enumerate(corr_matrix.columns):
        for j, sym2 in enumerate(corr_matrix.columns):
            if i < j:  # Only upper triangle
                corr = corr_matrix.loc[sym1, sym2]
                pair = (sym1, sym2, corr)

                if corr > 0.8:
                    high_corr.append(pair)
                elif corr < 0.3 and corr > 0:
                    low_corr.append(pair)
                elif corr < 0:
                    negative_corr.append(pair)

    result += "**Key Findings:**\n\n"

    if high_corr:
        result += "**🔗 Highly Correlated Pairs (>0.8):**\n"
        for sym1, sym2, corr in sorted(high_corr, key=lambda x: -x[2]):
            result += f"- {sym1} & {sym2}: {corr:.2f}\n"
            result += f"  *These move together - limited diversification benefit*\n"
        result += "\n"

    if low_corr:
        result += "**✨ Low Correlation Pairs (<0.3):**\n"
        for sym1, sym2, corr in sorted(low_corr, key=lambda x: x[2]):
            result += f"- {sym1} & {sym2}: {corr:.2f}\n"
            result += f"  *Good diversification - different risk drivers*\n"
        result += "\n"

    if negative_corr:
        result += "**🔄 Negatively Correlated Pairs:**\n"
        for sym1, sym2, corr in sorted(negative_corr, key=lambda x: x[2]):
            result += f"- {sym1} & {sym2}: {corr:.2f}\n"
            result += f"  *Natural hedge - tend to move opposite*\n"
        result += "\n"

    # Volatility comparison
    result += "**Volatility Comparison (Annualized):**\n"
    for sym in symbol_list:
        if sym in returns_df.columns:
            vol = returns_df[sym].std() * np.sqrt(252) * 100
            result += f"- {sym}: {vol:.1f}%\n"

    # Portfolio suggestions
    result += "\n**Diversification Insights:**\n"
    avg_corr = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()
    result += f"- Average pairwise correlation: {avg_corr:.2f}\n"

    if avg_corr > 0.7:
        result += "- ⚠️ High average correlation - consider adding uncorrelated assets\n"
    elif avg_corr < 0.4:
        result += "- ✅ Good diversification - assets have distinct risk profiles\n"
    else:
        result += "- Moderate diversification - some shared risk factors\n"

    return result


# =============================================================================
# Advanced Portfolio Management Tools (Norte Asset Management Quant Finance Guide)
# =============================================================================


