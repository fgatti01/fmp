"""News analysis tools for the financial agent.

This module provides tools for analyzing stock news, particularly around
earnings events, to understand the relationship between news and price movements.
"""

from datetime import datetime, timedelta
from typing import Any

from agno.tools import tool

from fmp_analytics.agent.tools._common import get_data_pipeline, run_async


@tool
def get_stock_news_tool(
    symbol: str,
    days_back: int = 7,
) -> str:
    """Get recent news articles for a stock.

    Fetches the latest news articles for a given stock symbol, including
    headlines, summaries, and publication dates.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT').
        days_back: Number of days to look back for news (default 7).

    Returns:
        Formatted string with recent news articles.
    """
    async def fetch():
        data_pipeline = get_data_pipeline()

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        news = await data_pipeline._company_info.get_stock_news(
            symbol=symbol.upper(),
            limit=20,
            from_date=from_date,
            to_date=to_date,
        )

        await data_pipeline._client.close()
        return news

    news = run_async(fetch())

    if not news:
        return f"No recent news found for {symbol.upper()}"

    result = f"## 📰 Recent News for {symbol.upper()}\n\n"

    for article in news[:10]:
        title = article.get("title", "No title")
        published = article.get("publishedDate", "")[:10]
        source = article.get("site", "Unknown")
        text = article.get("text", "")[:200]
        url = article.get("url", "")

        result += f"### {title}\n"
        result += f"📅 {published} | 📰 {source}\n"
        if text:
            result += f"> {text}...\n"
        if url:
            result += f"[Read more]({url})\n"
        result += "\n"

    return result


@tool
def analyze_earnings_news_impact_tool(
    symbol: str,
    quarters_back: int = 4,
) -> str:
    """Analyze the relationship between earnings news and price movements.

    This tool fetches historical earnings data and news around those dates,
    then analyzes whether the price movement was driven by:
    - Earnings beat/miss
    - Forward guidance
    - Other news catalysts
    - Technical factors

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL').
        quarters_back: Number of past quarters to analyze (default 4).

    Returns:
        Comprehensive analysis of earnings events and news impact.
    """
    async def fetch():
        data_pipeline = get_data_pipeline()

        # Get earnings surprises (historical earnings vs estimates)
        earnings = await data_pipeline._company_info.get_earnings_surprises(symbol.upper())

        # Get historical prices
        hist = await data_pipeline._historical.get_historical_price(
            symbol.upper(),
            time_series="daily"
        )

        # Get company profile
        profile = await data_pipeline._company_info.get_profile(symbol.upper())

        await data_pipeline._client.close()
        return earnings, hist, profile

    earnings, hist, profile = run_async(fetch())

    if not earnings:
        return f"No earnings data found for {symbol.upper()}"

    # Convert historical prices to lookup dict by date
    price_by_date = {}
    if hist and "historical" in hist:
        for day in hist["historical"]:
            price_by_date[day["date"]] = day

    company_name = ""
    if profile and len(profile) > 0:
        company_name = profile[0].get("companyName", symbol.upper())

    result = f"## 📊 Earnings News Impact Analysis: {company_name} ({symbol.upper()})\n\n"

    # Analyze each earnings event
    events_analyzed = []

    for earning in earnings[:quarters_back]:
        date = earning.get("date", "")
        actual_eps = earning.get("actualEarningResult", 0)
        estimated_eps = earning.get("estimatedEarning", 0)

        if not date or not actual_eps or not estimated_eps:
            continue

        # Calculate surprise
        surprise_pct = ((actual_eps - estimated_eps) / abs(estimated_eps)) * 100 if estimated_eps != 0 else 0

        # Get price data around earnings date
        earnings_date = datetime.strptime(date, "%Y-%m-%d")

        # Find prices around the earnings date
        pre_price = None
        post_price = None
        price_1d_after = None
        price_5d_after = None

        for i in range(-3, 0):
            check_date = (earnings_date + timedelta(days=i)).strftime("%Y-%m-%d")
            if check_date in price_by_date:
                pre_price = price_by_date[check_date]
                break

        for i in range(0, 4):
            check_date = (earnings_date + timedelta(days=i)).strftime("%Y-%m-%d")
            if check_date in price_by_date:
                if post_price is None:
                    post_price = price_by_date[check_date]
                    price_1d_after = price_by_date[check_date]
                break

        for i in range(4, 8):
            check_date = (earnings_date + timedelta(days=i)).strftime("%Y-%m-%d")
            if check_date in price_by_date:
                price_5d_after = price_by_date[check_date]
                break

        # Calculate price changes
        if pre_price and post_price:
            immediate_change = ((post_price["close"] - pre_price["close"]) / pre_price["close"]) * 100
            volume_ratio = post_price.get("volume", 0) / pre_price.get("volume", 1) if pre_price.get("volume", 0) > 0 else 1

            # 5-day follow through
            follow_through = None
            if price_5d_after:
                follow_through = ((price_5d_after["close"] - post_price["close"]) / post_price["close"]) * 100

            events_analyzed.append({
                "date": date,
                "actual_eps": actual_eps,
                "estimated_eps": estimated_eps,
                "surprise_pct": surprise_pct,
                "price_change": immediate_change,
                "volume_ratio": volume_ratio,
                "follow_through": follow_through,
                "pre_price": pre_price["close"],
                "post_price": post_price["close"],
            })

    if not events_analyzed:
        return f"Could not analyze earnings events for {symbol.upper()} - insufficient price data"

    # Summary statistics
    beats = [e for e in events_analyzed if e["surprise_pct"] > 2]
    misses = [e for e in events_analyzed if e["surprise_pct"] < -2]
    inline = [e for e in events_analyzed if -2 <= e["surprise_pct"] <= 2]

    result += "### 📈 Summary\n\n"
    result += f"- **Earnings Beats:** {len(beats)}\n"
    result += f"- **Earnings Misses:** {len(misses)}\n"
    result += f"- **In-Line Results:** {len(inline)}\n\n"

    # Analyze each event
    result += "### 🔍 Detailed Event Analysis\n\n"

    for event in events_analyzed:
        date = event["date"]
        surprise = event["surprise_pct"]
        price_change = event["price_change"]
        volume_ratio = event["volume_ratio"]
        follow_through = event["follow_through"]

        # Determine event type
        if surprise > 5:
            surprise_label = "🎯 Strong Beat"
        elif surprise > 2:
            surprise_label = "✅ Beat"
        elif surprise < -5:
            surprise_label = "❌ Strong Miss"
        elif surprise < -2:
            surprise_label = "⚠️ Miss"
        else:
            surprise_label = "➖ In-Line"

        # Price reaction emoji
        if price_change > 5:
            price_emoji = "🚀"
        elif price_change > 2:
            price_emoji = "📈"
        elif price_change < -5:
            price_emoji = "💥"
        elif price_change < -2:
            price_emoji = "📉"
        else:
            price_emoji = "➡️"

        result += f"#### {date}\n"
        result += f"- **EPS:** ${event['actual_eps']:.2f} vs ${event['estimated_eps']:.2f} estimate\n"
        result += f"- **Surprise:** {surprise_label} ({surprise:+.1f}%)\n"
        result += f"- **Price Reaction:** {price_emoji} {price_change:+.1f}%\n"
        result += f"- **Volume:** {volume_ratio:.1f}x normal\n"

        # Analyze the relationship
        result += "- **Analysis:** "

        # Check if price moved in expected direction
        if surprise > 2 and price_change > 2:
            result += "✅ *Earnings beat drove positive price action*\n"
        elif surprise > 2 and price_change < -2:
            result += "⚠️ *Beat but sold off - likely guidance concerns or 'sell the news'*\n"
        elif surprise < -2 and price_change < -2:
            result += "✅ *Earnings miss caused expected decline*\n"
        elif surprise < -2 and price_change > 2:
            result += "🔄 *Miss but rallied - low expectations or positive guidance*\n"
        elif abs(surprise) <= 2 and abs(price_change) > 3:
            result += "📊 *In-line results but significant move - guidance/outlook drove reaction*\n"
        else:
            result += "➖ *Muted reaction to results*\n"

        # Follow-through analysis
        if follow_through is not None:
            if price_change > 2 and follow_through > 0:
                result += f"- **Follow-Through:** 📈 Continued higher ({follow_through:+.1f}% over 5 days) - sustainable move\n"
            elif price_change > 2 and follow_through < -2:
                result += f"- **Follow-Through:** 🔄 Faded ({follow_through:+.1f}% over 5 days) - initial reaction overdone\n"
            elif price_change < -2 and follow_through < 0:
                result += f"- **Follow-Through:** 📉 Continued lower ({follow_through:+.1f}% over 5 days) - sustained selling\n"
            elif price_change < -2 and follow_through > 2:
                result += f"- **Follow-Through:** 🔄 Recovered ({follow_through:+.1f}% over 5 days) - initial selloff overdone\n"

        result += "\n"

    # Pattern recognition
    result += "### 🎯 Key Patterns Identified\n\n"

    # Calculate correlations
    surprise_prices = [(e["surprise_pct"], e["price_change"]) for e in events_analyzed]

    # Check if earnings surprise correlates with price
    beats_positive = len([e for e in events_analyzed if e["surprise_pct"] > 2 and e["price_change"] > 0])
    beats_total = len([e for e in events_analyzed if e["surprise_pct"] > 2])

    if beats_total > 0:
        beat_reaction_rate = beats_positive / beats_total * 100
        if beat_reaction_rate > 70:
            result += f"- 📈 **Reliable Beat Reaction:** {beat_reaction_rate:.0f}% of beats led to positive price action\n"
        elif beat_reaction_rate < 50:
            result += f"- ⚠️ **Unreliable Beat Reaction:** Only {beat_reaction_rate:.0f}% of beats led to gains (guidance matters more)\n"

    # High volume reactions
    high_vol_moves = [e for e in events_analyzed if e["volume_ratio"] > 2]
    if high_vol_moves:
        result += f"- 📊 **{len(high_vol_moves)} high-conviction moves** (>2x volume) in recent quarters\n"

    # Check for guidance sensitivity
    guidance_sensitive = [e for e in events_analyzed if abs(e["surprise_pct"]) < 3 and abs(e["price_change"]) > 5]
    if guidance_sensitive:
        result += f"- 🎤 **Guidance-sensitive stock:** {len(guidance_sensitive)} events with in-line EPS but large price moves\n"

    return result


@tool
def earnings_calendar_news_tool(
    days_ahead: int = 7,
) -> str:
    """Get upcoming earnings events with potential news catalysts.

    Fetches upcoming earnings announcements and identifies stocks
    that may have significant news-driven price movements.

    Args:
        days_ahead: Number of days to look ahead (default 7).

    Returns:
        Upcoming earnings events with context.
    """
    async def fetch():
        data_pipeline = get_data_pipeline()

        from_date = datetime.now().strftime("%Y-%m-%d")
        to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        earnings = await data_pipeline._calendar.get_earnings_calendar(
            from_date=from_date,
            to_date=to_date,
        )

        await data_pipeline._client.close()
        return earnings

    earnings = run_async(fetch())

    if not earnings:
        return f"No earnings scheduled in the next {days_ahead} days"

    result = f"## 📅 Upcoming Earnings (Next {days_ahead} Days)\n\n"
    result += "*Earnings events often drive significant price movements based on results vs expectations.*\n\n"

    # Group by date
    by_date: dict[str, list] = {}
    for e in earnings:
        date = e.get("date", "Unknown")
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(e)

    for date in sorted(by_date.keys())[:5]:
        result += f"### 📆 {date}\n\n"

        for e in by_date[date][:10]:
            symbol = e.get("symbol", "")
            eps_est = e.get("epsEstimated", "N/A")
            time = e.get("time", "")

            time_emoji = "🌅" if time == "bmo" else "🌙" if time == "amc" else "📍"
            time_label = "Before Market" if time == "bmo" else "After Market" if time == "amc" else time

            result += f"- **{symbol}** | EPS Est: ${eps_est} | {time_emoji} {time_label}\n"

        if len(by_date[date]) > 10:
            result += f"  *...and {len(by_date[date]) - 10} more*\n"

        result += "\n"

    result += "---\n"
    result += "*💡 Use `analyze_earnings_news_impact_tool` on specific stocks to see historical patterns*\n"

    return result


@tool
def news_sentiment_analysis_tool(
    symbol: str,
    days_back: int = 30,
) -> str:
    """Analyze news sentiment trends for a stock.

    Fetches recent news and analyzes sentiment patterns to identify
    whether the news flow is positive, negative, or neutral, and
    how it might be affecting the stock price.

    Args:
        symbol: Stock ticker symbol.
        days_back: Number of days to analyze (default 30).

    Returns:
        Sentiment analysis and trend insights.
    """
    async def fetch():
        data_pipeline = get_data_pipeline()

        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Get news
        news = await data_pipeline._company_info.get_stock_news(
            symbol=symbol.upper(),
            limit=50,
            from_date=from_date,
            to_date=to_date,
        )

        # Get price data for comparison
        hist = await data_pipeline._historical.get_historical_price(
            symbol.upper(),
            time_series="daily",
        )

        # Get profile
        profile = await data_pipeline._company_info.get_profile(symbol.upper())

        await data_pipeline._client.close()
        return news, hist, profile

    news, hist, profile = run_async(fetch())

    company_name = symbol.upper()
    if profile and len(profile) > 0:
        company_name = profile[0].get("companyName", symbol.upper())

    result = f"## 📰 News Sentiment Analysis: {company_name}\n\n"

    if not news:
        return f"No news found for {symbol.upper()} in the past {days_back} days"

    # Analyze news headlines using keyword-based sentiment
    positive_keywords = [
        "beat", "beats", "surpass", "exceed", "strong", "growth", "gain",
        "upgrade", "buy", "outperform", "bullish", "record", "soar", "surge",
        "profit", "revenue growth", "positive", "accelerat", "expand"
    ]
    negative_keywords = [
        "miss", "decline", "fall", "drop", "weak", "loss", "downgrade",
        "sell", "underperform", "bearish", "concern", "risk", "cut",
        "layoff", "restructur", "warning", "negative", "slump", "plunge"
    ]

    positive_count = 0
    negative_count = 0
    neutral_count = 0

    categorized_news = {"positive": [], "negative": [], "neutral": []}

    for article in news:
        title = article.get("title", "").lower()
        text = article.get("text", "").lower()
        content = title + " " + text

        pos_matches = sum(1 for kw in positive_keywords if kw in content)
        neg_matches = sum(1 for kw in negative_keywords if kw in content)

        if pos_matches > neg_matches:
            positive_count += 1
            categorized_news["positive"].append(article)
        elif neg_matches > pos_matches:
            negative_count += 1
            categorized_news["negative"].append(article)
        else:
            neutral_count += 1
            categorized_news["neutral"].append(article)

    total = positive_count + negative_count + neutral_count

    # Sentiment summary
    result += "### 📊 Sentiment Overview\n\n"

    pos_pct = positive_count / total * 100 if total > 0 else 0
    neg_pct = negative_count / total * 100 if total > 0 else 0
    neu_pct = neutral_count / total * 100 if total > 0 else 0

    result += f"| Sentiment | Count | Percentage |\n"
    result += f"|-----------|-------|------------|\n"
    result += f"| 🟢 Positive | {positive_count} | {pos_pct:.0f}% |\n"
    result += f"| 🔴 Negative | {negative_count} | {neg_pct:.0f}% |\n"
    result += f"| ⚪ Neutral | {neutral_count} | {neu_pct:.0f}% |\n\n"

    # Overall sentiment
    if pos_pct - neg_pct > 20:
        result += "**Overall Sentiment:** 📈 **Bullish** - Positive news flow dominates\n\n"
    elif neg_pct - pos_pct > 20:
        result += "**Overall Sentiment:** 📉 **Bearish** - Negative news flow dominates\n\n"
    else:
        result += "**Overall Sentiment:** ➡️ **Mixed/Neutral** - Balanced news flow\n\n"

    # Sample headlines by sentiment
    if categorized_news["positive"]:
        result += "### 🟢 Positive Headlines\n"
        for article in categorized_news["positive"][:3]:
            title = article.get("title", "")
            date = article.get("publishedDate", "")[:10]
            result += f"- *{date}:* {title}\n"
        result += "\n"

    if categorized_news["negative"]:
        result += "### 🔴 Negative Headlines\n"
        for article in categorized_news["negative"][:3]:
            title = article.get("title", "")
            date = article.get("publishedDate", "")[:10]
            result += f"- *{date}:* {title}\n"
        result += "\n"

    # Price correlation
    if hist and "historical" in hist:
        prices = hist["historical"][:days_back]
        if len(prices) >= 2:
            start_price = prices[-1]["close"]
            end_price = prices[0]["close"]
            price_change = ((end_price - start_price) / start_price) * 100

            result += "### 📈 Price vs Sentiment\n\n"
            result += f"- **{days_back}-Day Price Change:** {price_change:+.1f}%\n"

            if price_change > 5 and pos_pct > neg_pct:
                result += "- **Correlation:** ✅ Positive news flow aligned with price gains\n"
            elif price_change < -5 and neg_pct > pos_pct:
                result += "- **Correlation:** ✅ Negative news flow aligned with price decline\n"
            elif price_change > 5 and neg_pct > pos_pct:
                result += "- **Divergence:** ⚠️ Price rising despite negative news - possible turnaround\n"
            elif price_change < -5 and pos_pct > neg_pct:
                result += "- **Divergence:** ⚠️ Price falling despite positive news - underlying concerns\n"
            else:
                result += "- **Correlation:** ➖ News and price movement are balanced\n"

    return result


# Export all tools
__all__ = [
    "get_stock_news_tool",
    "analyze_earnings_news_impact_tool",
    "earnings_calendar_news_tool",
    "news_sentiment_analysis_tool",
]
