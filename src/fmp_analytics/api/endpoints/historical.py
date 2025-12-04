"""Historical price data endpoints."""

from typing import Any

from fmp_analytics.api.endpoints.base import BaseAPI


class HistoricalDataAPI(BaseAPI):
    """API endpoints for historical price data."""

    async def get_historical_price(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Get daily historical prices.

        Args:
            symbol: Stock symbol.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Historical price data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get(f"historical-price-full/{symbol}", params=params)

    async def get_historical_price_batch(
        self,
        symbols: list[str],
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Get historical prices for multiple symbols.

        Args:
            symbols: List of stock symbols.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Historical price data for all symbols.
        """
        symbols_str = ",".join(symbols)
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get(f"historical-price-full/{symbols_str}", params=params)

    async def get_intraday_prices(
        self,
        symbol: str,
        interval: str = "1min",
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get intraday historical prices.

        Args:
            symbol: Stock symbol.
            interval: Time interval (1min, 5min, 15min, 30min, 1hour, 4hour).
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Intraday price data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get(f"historical-chart/{interval}/{symbol}", params=params)

    async def get_daily_line(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get daily closing prices (lightweight).

        Args:
            symbol: Stock symbol.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Daily closing prices.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get(f"historical-price-full/{symbol}", params={**params, "serietype": "line"})

    async def get_stock_dividend_history(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """Get historical dividend data.

        Args:
            symbol: Stock symbol.

        Returns:
            Dividend history.
        """
        return await self._client.get(f"historical-price-full/stock_dividend/{symbol}")

    async def get_stock_split_history(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """Get historical stock split data.

        Args:
            symbol: Stock symbol.

        Returns:
            Stock split history.
        """
        return await self._client.get(f"historical-price-full/stock_split/{symbol}")

    async def get_survivorship_bias_free(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get survivorship bias-free historical data.

        Args:
            symbol: Stock symbol.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Survivorship bias-free data.
        """
        params: dict[str, Any] = {"symbol": symbol}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get_v4("historical-price-full", params=params)

    async def get_forex_historical(
        self,
        pair: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Get historical forex prices.

        Args:
            pair: Currency pair (e.g., EURUSD).
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Historical forex data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get(f"historical-price-full/{pair}", params=params)

    async def get_crypto_historical(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Get historical cryptocurrency prices.

        Args:
            symbol: Crypto symbol (e.g., BTCUSD).
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Historical crypto data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get(f"historical-price-full/{symbol}", params=params)

    async def get_index_historical(
        self,
        symbol: str,
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        """Get historical index prices.

        Args:
            symbol: Index symbol.
            from_date: Start date (YYYY-MM-DD).
            to_date: End date (YYYY-MM-DD).

        Returns:
            Historical index data.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        return await self._client.get(f"historical-price-full/{symbol}", params=params)

    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        period: int = 10,
        interval: str = "daily",
    ) -> list[dict[str, Any]]:
        """Get technical indicator data.

        Args:
            symbol: Stock symbol.
            indicator: Indicator type (sma, ema, wma, dema, tema, williams, rsi, adx, standardDeviation).
            period: Indicator period.
            interval: Time interval (1min, 5min, 15min, 30min, 1hour, 4hour, daily).

        Returns:
            Technical indicator data.
        """
        return await self._client.get(
            f"technical_indicator/{interval}/{symbol}",
            params={"period": period, "type": indicator},
        )

    async def get_daily_indicators(
        self,
        symbol: str,
        period: int = 10,
        indicator: str = "sma",
    ) -> list[dict[str, Any]]:
        """Get daily technical indicator.

        Args:
            symbol: Stock symbol.
            period: Indicator period.
            indicator: Indicator type.

        Returns:
            Daily indicator data.
        """
        return await self.get_technical_indicator(symbol, indicator, period, "daily")

    async def get_full_historical_price(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """Get full historical price data (max available).

        Args:
            symbol: Stock symbol.

        Returns:
            Full historical data.
        """
        return await self._client.get(f"historical-price-full/{symbol}")
