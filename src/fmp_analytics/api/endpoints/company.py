"""Company information and profile endpoints."""

from typing import Any

from fmp_analytics.api.endpoints.base import BaseAPI


class CompanyInfoAPI(BaseAPI):
    """API endpoints for company information and profiles."""

    async def get_profile(self, symbol: str) -> list[dict[str, Any]]:
        """Get company profile.

        Args:
            symbol: Stock symbol.

        Returns:
            Company profile data.
        """
        return await self._client.get(f"profile/{symbol}")

    async def get_profiles_batch(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Get profiles for multiple companies.

        Args:
            symbols: List of stock symbols.

        Returns:
            List of company profiles.
        """
        symbols_str = ",".join(symbols)
        return await self._client.get(f"profile/{symbols_str}")

    async def get_key_executives(self, symbol: str) -> list[dict[str, Any]]:
        """Get key executives for a company.

        Args:
            symbol: Stock symbol.

        Returns:
            List of key executives.
        """
        return await self._client.get(f"key-executives/{symbol}")

    async def get_company_outlook(self, symbol: str) -> dict[str, Any]:
        """Get comprehensive company outlook.

        Args:
            symbol: Stock symbol.

        Returns:
            Company outlook with metrics, ratios, and data.
        """
        return await self._client.get_v4("company-outlook", params={"symbol": symbol})

    async def get_stock_peers(self, symbol: str) -> list[dict[str, Any]]:
        """Get peer companies.

        Args:
            symbol: Stock symbol.

        Returns:
            List of peer companies.
        """
        return await self._client.get_v4("stock_peers", params={"symbol": symbol})

    async def get_core_information(self, symbol: str) -> list[dict[str, Any]]:
        """Get core company information.

        Args:
            symbol: Stock symbol.

        Returns:
            Core company information.
        """
        return await self._client.get_v4("company-core-information", params={"symbol": symbol})

    async def get_market_cap(self, symbol: str) -> list[dict[str, Any]]:
        """Get current market capitalization.

        Args:
            symbol: Stock symbol.

        Returns:
            Market cap data.
        """
        return await self._client.get(f"market-capitalization/{symbol}")

    async def get_historical_market_cap(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get historical market capitalization.

        Args:
            symbol: Stock symbol.
            limit: Number of records.

        Returns:
            Historical market cap data.
        """
        return await self._client.get(
            f"historical-market-capitalization/{symbol}",
            params={"limit": limit},
        )

    async def get_company_rating(self, symbol: str) -> list[dict[str, Any]]:
        """Get company rating.

        Args:
            symbol: Stock symbol.

        Returns:
            Company rating data.
        """
        return await self._client.get(f"rating/{symbol}")

    async def get_historical_rating(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get historical company ratings.

        Args:
            symbol: Stock symbol.
            limit: Number of records.

        Returns:
            Historical rating data.
        """
        return await self._client.get(f"historical-rating/{symbol}", params={"limit": limit})

    async def get_analyst_estimates(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Get analyst estimates.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of records.

        Returns:
            Analyst estimates data.
        """
        return await self._client.get(
            f"analyst-estimates/{symbol}",
            params={"period": period, "limit": limit},
        )

    async def get_analyst_recommendations(self, symbol: str) -> list[dict[str, Any]]:
        """Get analyst recommendations.

        Args:
            symbol: Stock symbol.

        Returns:
            Analyst recommendations.
        """
        return await self._client.get(f"analyst-stock-recommendations/{symbol}")

    async def get_price_target(self, symbol: str) -> list[dict[str, Any]]:
        """Get analyst price targets.

        Args:
            symbol: Stock symbol.

        Returns:
            Price target data.
        """
        return await self._client.get_v4("price-target", params={"symbol": symbol})

    async def get_price_target_summary(self, symbol: str) -> list[dict[str, Any]]:
        """Get price target summary.

        Args:
            symbol: Stock symbol.

        Returns:
            Price target summary.
        """
        return await self._client.get_v4("price-target-summary", params={"symbol": symbol})

    async def get_price_target_consensus(self, symbol: str) -> list[dict[str, Any]]:
        """Get consensus price target.

        Args:
            symbol: Stock symbol.

        Returns:
            Consensus price target.
        """
        return await self._client.get_v4("price-target-consensus", params={"symbol": symbol})

    async def get_grade(self, symbol: str, limit: int = 500) -> list[dict[str, Any]]:
        """Get company grade/rating history.

        Args:
            symbol: Stock symbol.
            limit: Number of records.

        Returns:
            Grade history.
        """
        return await self._client.get(f"grade/{symbol}", params={"limit": limit})

    async def get_earnings_surprises(self, symbol: str) -> list[dict[str, Any]]:
        """Get earnings surprises.

        Args:
            symbol: Stock symbol.

        Returns:
            Earnings surprises data.
        """
        return await self._client.get(f"earnings-surprises/{symbol}")

    async def get_sec_filings(
        self,
        symbol: str,
        filing_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get SEC filings.

        Args:
            symbol: Stock symbol.
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.).
            limit: Number of records.

        Returns:
            SEC filings data.
        """
        params: dict[str, Any] = {"limit": limit}
        if filing_type:
            params["type"] = filing_type
        return await self._client.get(f"sec_filings/{symbol}", params=params)

    async def get_press_releases(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get press releases.

        Args:
            symbol: Stock symbol.
            limit: Number of records.

        Returns:
            Press releases.
        """
        return await self._client.get(f"press-releases/{symbol}", params={"limit": limit})

    async def search(
        self,
        query: str,
        limit: int = 10,
        exchange: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for companies.

        Args:
            query: Search query.
            limit: Maximum results.
            exchange: Filter by exchange.

        Returns:
            Search results.
        """
        params: dict[str, Any] = {"query": query, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        return await self._client.get("search", params=params)

    async def search_ticker(
        self,
        query: str,
        limit: int = 10,
        exchange: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for tickers.

        Args:
            query: Search query.
            limit: Maximum results.
            exchange: Filter by exchange.

        Returns:
            Ticker search results.
        """
        params: dict[str, Any] = {"query": query, "limit": limit}
        if exchange:
            params["exchange"] = exchange
        return await self._client.get("search-ticker", params=params)

    # ==================== NEWS ENDPOINTS ====================

    async def get_stock_news(
        self,
        symbol: str | None = None,
        limit: int = 50,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get stock news articles.

        Args:
            symbol: Stock symbol (optional for general news).
            limit: Number of articles.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            List of news articles with title, text, url, publishedDate, sentiment.
        """
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["tickers"] = symbol.upper()
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get("stock_news", params=params)

    async def get_general_news(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get general market news.

        Args:
            limit: Number of articles.

        Returns:
            List of general news articles.
        """
        return await self._client.get_v4("general_news", params={"limit": limit})

    async def get_forex_news(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get forex market news.

        Args:
            limit: Number of articles.

        Returns:
            List of forex news articles.
        """
        return await self._client.get_v4("forex_news", params={"limit": limit})

    async def get_crypto_news(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get cryptocurrency news.

        Args:
            limit: Number of articles.

        Returns:
            List of crypto news articles.
        """
        return await self._client.get_v4("crypto_news", params={"limit": limit})

    async def get_stock_news_sentiment(
        self,
        symbol: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get stock news with sentiment analysis.

        Args:
            symbol: Stock symbol.
            limit: Number of articles.

        Returns:
            News articles with sentiment scores.
        """
        return await self._client.get_v4(
            "stock-news-sentiments-rss-feed",
            params={"symbol": symbol.upper(), "limit": limit},
        )
