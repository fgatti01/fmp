"""Market data and quotes endpoints."""

from typing import Any

from fmp_analytics.api.endpoints.base import BaseAPI


class MarketDataAPI(BaseAPI):
    """API endpoints for market data and quotes."""

    async def get_quote(self, symbol: str) -> list[dict[str, Any]]:
        """Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Quote data.
        """
        return await self._client.get(f"quote/{symbol}")

    async def get_quotes_batch(self, symbols: list[str]) -> list[dict[str, Any]]:
        """Get quotes for multiple symbols.

        Args:
            symbols: List of stock symbols.

        Returns:
            List of quotes.
        """
        symbols_str = ",".join(symbols)
        return await self._client.get(f"quote/{symbols_str}")

    async def get_quote_short(self, symbol: str) -> list[dict[str, Any]]:
        """Get short quote (price only).

        Args:
            symbol: Stock symbol.

        Returns:
            Short quote data.
        """
        return await self._client.get(f"quote-short/{symbol}")

    async def get_otc_quote(self, symbol: str) -> list[dict[str, Any]]:
        """Get OTC market quote.

        Args:
            symbol: Stock symbol.

        Returns:
            OTC quote data.
        """
        return await self._client.get(f"otc/real-time-price/{symbol}")

    async def get_exchange_prices(self, exchange: str) -> list[dict[str, Any]]:
        """Get all prices for an exchange.

        Args:
            exchange: Exchange code.

        Returns:
            All prices on the exchange.
        """
        return await self._client.get(f"quotes/{exchange}")

    async def get_all_prices(self) -> list[dict[str, Any]]:
        """Get all stock prices.

        Returns:
            All available stock prices.
        """
        return await self._client.get("quotes/index")

    async def get_price_change(self, symbol: str) -> list[dict[str, Any]]:
        """Get stock price change.

        Args:
            symbol: Stock symbol.

        Returns:
            Price change data.
        """
        return await self._client.get(f"stock-price-change/{symbol}")

    async def get_gainers(self) -> list[dict[str, Any]]:
        """Get top gaining stocks.

        Returns:
            List of top gainers.
        """
        return await self._client.get("stock_market/gainers")

    async def get_losers(self) -> list[dict[str, Any]]:
        """Get top losing stocks.

        Returns:
            List of top losers.
        """
        return await self._client.get("stock_market/losers")

    async def get_most_active(self) -> list[dict[str, Any]]:
        """Get most active stocks.

        Returns:
            List of most active stocks.
        """
        return await self._client.get("stock_market/actives")

    async def get_sector_performance(self) -> list[dict[str, Any]]:
        """Get sector performance.

        Returns:
            Sector performance data.
        """
        return await self._client.get("sector-performance")

    async def get_historical_sector_performance(
        self,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get historical sector performance.

        Args:
            limit: Number of records.

        Returns:
            Historical sector performance.
        """
        return await self._client.get("historical-sectors-performance", params={"limit": limit})

    async def get_market_hours(self) -> dict[str, Any]:
        """Get market hours status.

        Returns:
            Market hours data.
        """
        return await self._client.get("market-hours")

    async def is_market_open(self, exchange: str = "NYSE") -> dict[str, Any]:
        """Check if market is open.

        Args:
            exchange: Exchange to check.

        Returns:
            Market open status.
        """
        return await self._client.get(f"is-the-market-open/{exchange}")

    async def get_after_hours_quote(self, symbol: str) -> list[dict[str, Any]]:
        """Get after-hours trading quote.

        Args:
            symbol: Stock symbol.

        Returns:
            After-hours quote data.
        """
        return await self._client.get_v4("pre-post-market-trade", params={"symbol": symbol})

    async def get_forex_list(self) -> list[dict[str, Any]]:
        """Get all available forex pairs.

        Returns:
            List of forex pairs.
        """
        return await self._client.get("symbol/available-forex-currency-pairs")

    async def get_forex_quote(self, pair: str) -> list[dict[str, Any]]:
        """Get forex quote.

        Args:
            pair: Currency pair (e.g., EURUSD).

        Returns:
            Forex quote data.
        """
        return await self._client.get(f"fx/{pair}")

    async def get_all_forex_quotes(self) -> list[dict[str, Any]]:
        """Get all forex quotes.

        Returns:
            All forex quotes.
        """
        return await self._client.get("fx")

    async def get_crypto_list(self) -> list[dict[str, Any]]:
        """Get all available cryptocurrencies.

        Returns:
            List of cryptocurrencies.
        """
        return await self._client.get("symbol/available-cryptocurrencies")

    async def get_crypto_quote(self, symbol: str) -> list[dict[str, Any]]:
        """Get cryptocurrency quote.

        Args:
            symbol: Crypto symbol (e.g., BTCUSD).

        Returns:
            Crypto quote data.
        """
        return await self._client.get(f"quote/{symbol}")

    async def get_all_crypto_quotes(self) -> list[dict[str, Any]]:
        """Get all cryptocurrency quotes.

        Returns:
            All crypto quotes.
        """
        return await self._client.get("quotes/crypto")

    async def get_commodities_list(self) -> list[dict[str, Any]]:
        """Get all available commodities.

        Returns:
            List of commodities.
        """
        return await self._client.get("symbol/available-commodities")

    async def get_commodity_quote(self, symbol: str) -> list[dict[str, Any]]:
        """Get commodity quote.

        Args:
            symbol: Commodity symbol.

        Returns:
            Commodity quote data.
        """
        return await self._client.get(f"quote/{symbol}")

    async def get_all_commodity_quotes(self) -> list[dict[str, Any]]:
        """Get all commodity quotes.

        Returns:
            All commodity quotes.
        """
        return await self._client.get("quotes/commodity")

    async def get_index_list(self) -> list[dict[str, Any]]:
        """Get all available indices.

        Returns:
            List of indices.
        """
        return await self._client.get("symbol/available-indexes")

    async def get_index_quote(self, symbol: str) -> list[dict[str, Any]]:
        """Get index quote.

        Args:
            symbol: Index symbol.

        Returns:
            Index quote data.
        """
        return await self._client.get(f"quote/{symbol}")

    async def get_all_index_quotes(self) -> list[dict[str, Any]]:
        """Get all index quotes.

        Returns:
            All index quotes.
        """
        return await self._client.get("quotes/index")

    async def get_mutual_fund_list(self) -> list[dict[str, Any]]:
        """Get all mutual funds.

        Returns:
            List of mutual funds.
        """
        return await self._client.get("symbol/available-mutual-funds")

    async def get_etf_quote(self, symbol: str) -> list[dict[str, Any]]:
        """Get ETF quote.

        Args:
            symbol: ETF symbol.

        Returns:
            ETF quote data.
        """
        return await self._client.get(f"quote/{symbol}")

    async def get_all_etf_quotes(self) -> list[dict[str, Any]]:
        """Get all ETF quotes.

        Returns:
            All ETF quotes.
        """
        return await self._client.get("quotes/etf")
