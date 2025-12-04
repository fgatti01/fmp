"""Stock list and symbol endpoints."""

from typing import Any

from fmp_analytics.api.endpoints.base import BaseAPI


class StockListAPI(BaseAPI):
    """API endpoints for stock lists and symbols."""

    async def get_stock_list(self) -> list[dict[str, Any]]:
        """Get all available stock symbols.

        Returns:
            List of stock symbols with basic info.
        """
        return await self._client.get("stock/list")

    async def get_tradable_list(self) -> list[dict[str, Any]]:
        """Get all tradable symbols.

        Returns:
            List of tradable symbols.
        """
        return await self._client.get("available-traded/list")

    async def get_etf_list(self) -> list[dict[str, Any]]:
        """Get all ETF symbols.

        Returns:
            List of ETF symbols.
        """
        return await self._client.get("etf/list")

    async def get_stock_screener(
        self,
        market_cap_more_than: int | None = None,
        market_cap_less_than: int | None = None,
        price_more_than: float | None = None,
        price_less_than: float | None = None,
        beta_more_than: float | None = None,
        beta_less_than: float | None = None,
        volume_more_than: int | None = None,
        volume_less_than: int | None = None,
        dividend_more_than: float | None = None,
        dividend_less_than: float | None = None,
        is_etf: bool | None = None,
        is_actively_trading: bool | None = None,
        sector: str | None = None,
        industry: str | None = None,
        country: str | None = None,
        exchange: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Screen stocks based on various criteria.

        Args:
            market_cap_more_than: Minimum market cap.
            market_cap_less_than: Maximum market cap.
            price_more_than: Minimum price.
            price_less_than: Maximum price.
            beta_more_than: Minimum beta.
            beta_less_than: Maximum beta.
            volume_more_than: Minimum volume.
            volume_less_than: Maximum volume.
            dividend_more_than: Minimum dividend yield.
            dividend_less_than: Maximum dividend yield.
            is_etf: Filter for ETFs only.
            is_actively_trading: Filter for actively trading.
            sector: Filter by sector.
            industry: Filter by industry.
            country: Filter by country.
            exchange: Filter by exchange.
            limit: Maximum results to return.

        Returns:
            List of stocks matching criteria.
        """
        params: dict[str, Any] = {"limit": limit}

        if market_cap_more_than is not None:
            params["marketCapMoreThan"] = market_cap_more_than
        if market_cap_less_than is not None:
            params["marketCapLowerThan"] = market_cap_less_than
        if price_more_than is not None:
            params["priceMoreThan"] = price_more_than
        if price_less_than is not None:
            params["priceLowerThan"] = price_less_than
        if beta_more_than is not None:
            params["betaMoreThan"] = beta_more_than
        if beta_less_than is not None:
            params["betaLowerThan"] = beta_less_than
        if volume_more_than is not None:
            params["volumeMoreThan"] = volume_more_than
        if volume_less_than is not None:
            params["volumeLowerThan"] = volume_less_than
        if dividend_more_than is not None:
            params["dividendMoreThan"] = dividend_more_than
        if dividend_less_than is not None:
            params["dividendLowerThan"] = dividend_less_than
        if is_etf is not None:
            params["isEtf"] = str(is_etf).lower()
        if is_actively_trading is not None:
            params["isActivelyTrading"] = str(is_actively_trading).lower()
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry
        if country:
            params["country"] = country
        if exchange:
            params["exchange"] = exchange

        return await self._client.get("stock-screener", params=params)

    async def get_symbols_by_exchange(self, exchange: str) -> list[dict[str, Any]]:
        """Get all symbols for a specific exchange.

        Args:
            exchange: Exchange code (e.g., NYSE, NASDAQ).

        Returns:
            List of symbols on the exchange.
        """
        return await self._client.get(f"symbol/{exchange}")

    async def get_cik_list(self) -> list[dict[str, Any]]:
        """Get CIK (Central Index Key) list for SEC filings.

        Returns:
            List of CIK numbers and company info.
        """
        return await self._client.get("cik_list")

    async def get_commitment_of_traders(self, symbol: str) -> list[dict[str, Any]]:
        """Get Commitment of Traders report data.

        Args:
            symbol: Trading symbol.

        Returns:
            COT report data.
        """
        return await self._client.get_v4("commitment_of_traders_report", params={"symbol": symbol})

    async def get_sp500_constituents(self) -> list[dict[str, Any]]:
        """Get S&P 500 index constituents.

        Returns:
            List of S&P 500 companies.
        """
        return await self._client.get("sp500_constituent")

    async def get_nasdaq_constituents(self) -> list[dict[str, Any]]:
        """Get NASDAQ 100 index constituents.

        Returns:
            List of NASDAQ 100 companies.
        """
        return await self._client.get("nasdaq_constituent")

    async def get_dowjones_constituents(self) -> list[dict[str, Any]]:
        """Get Dow Jones index constituents.

        Returns:
            List of Dow Jones companies.
        """
        return await self._client.get("dowjones_constituent")
