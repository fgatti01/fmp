"""Calendar and events endpoints."""

from typing import Any

from fmp_analytics.api.endpoints.base import BaseAPI


class CalendarAPI(BaseAPI):
    """API endpoints for financial calendars and events."""

    async def get_earnings_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get earnings calendar.

        Args:
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Earnings calendar data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get("earning_calendar", params=params)

    async def get_earnings_confirmed(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get confirmed earnings dates.

        Args:
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Confirmed earnings data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get_v4("earning-calendar-confirmed", params=params)

    async def get_ipo_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get IPO calendar.

        Args:
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            IPO calendar data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get("ipo_calendar", params=params)

    async def get_ipo_confirmed(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get confirmed IPO dates.

        Args:
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Confirmed IPO data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get_v4("ipo-calendar-confirmed", params=params)

    async def get_stock_split_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get stock split calendar.

        Args:
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Stock split calendar data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get("stock_split_calendar", params=params)

    async def get_dividend_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get dividend calendar.

        Args:
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Dividend calendar data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get("stock_dividend_calendar", params=params)

    async def get_economic_calendar(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get economic events calendar.

        Args:
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Economic calendar data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get("economic_calendar", params=params)

    async def get_historical_sp500_constituents(self) -> list[dict[str, Any]]:
        """Get historical S&P 500 constituent changes.

        Returns:
            Historical constituent changes.
        """
        return await self._client.get("historical/sp500_constituent")

    async def get_historical_nasdaq_constituents(self) -> list[dict[str, Any]]:
        """Get historical NASDAQ 100 constituent changes.

        Returns:
            Historical constituent changes.
        """
        return await self._client.get("historical/nasdaq_constituent")

    async def get_historical_dowjones_constituents(self) -> list[dict[str, Any]]:
        """Get historical Dow Jones constituent changes.

        Returns:
            Historical constituent changes.
        """
        return await self._client.get("historical/dowjones_constituent")

    async def get_market_holidays(
        self,
        exchange: str = "NYSE",
    ) -> list[dict[str, Any]]:
        """Get market holidays.

        Args:
            exchange: Exchange code.

        Returns:
            Market holidays data.
        """
        return await self._client.get(f"is-the-market-open/{exchange}")

    async def get_treasury_rates(
        self,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get treasury rates.

        Args:
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Treasury rates data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get_v4("treasury", params=params)

    async def get_mergers_acquisitions(
        self,
        page: int = 0,
    ) -> list[dict[str, Any]]:
        """Get mergers and acquisitions.

        Args:
            page: Page number.

        Returns:
            M&A data.
        """
        return await self._client.get_v4("mergers-acquisitions-rss-feed", params={"page": page})

    async def get_sec_rss_feed(
        self,
        limit: int = 50,
        filing_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get SEC RSS feed.

        Args:
            limit: Number of records.
            filing_type: Type of filing.

        Returns:
            SEC RSS feed data.
        """
        params: dict[str, Any] = {"limit": limit}
        if filing_type:
            params["type"] = filing_type
        return await self._client.get_v4("rss_feed", params=params)

    async def get_crowdfunding_rss(
        self,
        page: int = 0,
    ) -> list[dict[str, Any]]:
        """Get crowdfunding offerings RSS feed.

        Args:
            page: Page number.

        Returns:
            Crowdfunding RSS data.
        """
        return await self._client.get_v4("crowdfunding-offerings-rss-feed", params={"page": page})

    async def get_fundraising_rss(
        self,
        page: int = 0,
    ) -> list[dict[str, Any]]:
        """Get equity offerings RSS feed.

        Args:
            page: Page number.

        Returns:
            Equity offerings RSS data.
        """
        return await self._client.get_v4("fundraising-rss-feed", params={"page": page})
