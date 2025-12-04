"""Financial statements and metrics endpoints."""

from typing import Any

from fmp_analytics.api.endpoints.base import BaseAPI


class FinancialStatementsAPI(BaseAPI):
    """API endpoints for financial statements and metrics."""

    # Income Statements
    async def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get income statements.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Income statement data.
        """
        return await self._client.get(
            f"income-statement/{symbol}",
            params={"period": period, "limit": limit},
        )

    async def get_income_statement_growth(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get income statement growth metrics.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Income statement growth data.
        """
        return await self._client.get(
            f"income-statement-growth/{symbol}",
            params={"period": period, "limit": limit},
        )

    # Balance Sheets
    async def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get balance sheet statements.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Balance sheet data.
        """
        return await self._client.get(
            f"balance-sheet-statement/{symbol}",
            params={"period": period, "limit": limit},
        )

    async def get_balance_sheet_growth(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get balance sheet growth metrics.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Balance sheet growth data.
        """
        return await self._client.get(
            f"balance-sheet-statement-growth/{symbol}",
            params={"period": period, "limit": limit},
        )

    # Cash Flow Statements
    async def get_cash_flow_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get cash flow statements.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Cash flow statement data.
        """
        return await self._client.get(
            f"cash-flow-statement/{symbol}",
            params={"period": period, "limit": limit},
        )

    async def get_cash_flow_growth(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get cash flow growth metrics.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Cash flow growth data.
        """
        return await self._client.get(
            f"cash-flow-statement-growth/{symbol}",
            params={"period": period, "limit": limit},
        )

    # Financial Ratios
    async def get_ratios(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get financial ratios.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Financial ratios data.
        """
        return await self._client.get(
            f"ratios/{symbol}",
            params={"period": period, "limit": limit},
        )

    async def get_ratios_ttm(self, symbol: str) -> list[dict[str, Any]]:
        """Get trailing twelve month ratios.

        Args:
            symbol: Stock symbol.

        Returns:
            TTM ratios data.
        """
        return await self._client.get(f"ratios-ttm/{symbol}")

    # Key Metrics
    async def get_key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get key financial metrics.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Key metrics data.
        """
        return await self._client.get(
            f"key-metrics/{symbol}",
            params={"period": period, "limit": limit},
        )

    async def get_key_metrics_ttm(self, symbol: str) -> list[dict[str, Any]]:
        """Get trailing twelve month key metrics.

        Args:
            symbol: Stock symbol.

        Returns:
            TTM key metrics data.
        """
        return await self._client.get(f"key-metrics-ttm/{symbol}")

    # Financial Growth
    async def get_financial_growth(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get financial growth metrics.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Financial growth data.
        """
        return await self._client.get(
            f"financial-growth/{symbol}",
            params={"period": period, "limit": limit},
        )

    # Enterprise Value
    async def get_enterprise_value(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get enterprise value data.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Enterprise value data.
        """
        return await self._client.get(
            f"enterprise-values/{symbol}",
            params={"period": period, "limit": limit},
        )

    # Discounted Cash Flow
    async def get_dcf(self, symbol: str) -> list[dict[str, Any]]:
        """Get discounted cash flow valuation.

        Args:
            symbol: Stock symbol.

        Returns:
            DCF valuation data.
        """
        return await self._client.get(f"discounted-cash-flow/{symbol}")

    async def get_historical_dcf(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get historical DCF valuations.

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Historical DCF data.
        """
        return await self._client.get(
            f"historical-discounted-cash-flow-statement/{symbol}",
            params={"period": period, "limit": limit},
        )

    async def get_advanced_dcf(self, symbol: str) -> dict[str, Any]:
        """Get advanced DCF analysis.

        Args:
            symbol: Stock symbol.

        Returns:
            Advanced DCF data.
        """
        return await self._client.get_v4("advanced_discounted_cash_flow", params={"symbol": symbol})

    async def get_levered_dcf(self, symbol: str) -> dict[str, Any]:
        """Get levered DCF analysis.

        Args:
            symbol: Stock symbol.

        Returns:
            Levered DCF data.
        """
        return await self._client.get_v4("advanced_levered_discounted_cash_flow", params={"symbol": symbol})

    # Owner Earnings
    async def get_owner_earnings(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get owner earnings (Buffett method).

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).
            limit: Number of periods.

        Returns:
            Owner earnings data.
        """
        return await self._client.get_v4(
            "owner_earnings",
            params={"symbol": symbol, "period": period, "limit": limit},
        )

    # Financial Score
    async def get_financial_score(self, symbol: str) -> list[dict[str, Any]]:
        """Get financial health score.

        Args:
            symbol: Stock symbol.

        Returns:
            Financial score data.
        """
        return await self._client.get_v4("score", params={"symbol": symbol})

    # Shares Float
    async def get_shares_float(self, symbol: str) -> list[dict[str, Any]]:
        """Get shares float information.

        Args:
            symbol: Stock symbol.

        Returns:
            Shares float data.
        """
        return await self._client.get_v4("shares_float", params={"symbol": symbol})

    # Earnings
    async def get_earnings_historical(
        self,
        symbol: str,
        limit: int = 80,
    ) -> list[dict[str, Any]]:
        """Get historical earnings data.

        Args:
            symbol: Stock symbol.
            limit: Number of periods.

        Returns:
            Historical earnings data.
        """
        return await self._client.get(f"historical/earning_calendar/{symbol}", params={"limit": limit})

    # Full Financial Statement
    async def get_full_financial_statement(
        self,
        symbol: str,
        period: str = "annual",
    ) -> list[dict[str, Any]]:
        """Get full financial statement (all three statements).

        Args:
            symbol: Stock symbol.
            period: Period type (annual or quarter).

        Returns:
            Full financial statement data.
        """
        return await self._client.get(
            f"financial-statement-full-as-reported/{symbol}",
            params={"period": period},
        )

    # Insider Trading
    async def get_insider_trading(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get insider trading data.

        Args:
            symbol: Stock symbol.
            limit: Number of records.

        Returns:
            Insider trading data.
        """
        return await self._client.get_v4("insider-trading", params={"symbol": symbol, "limit": limit})

    # Institutional Holders
    async def get_institutional_holders(
        self,
        symbol: str,
    ) -> list[dict[str, Any]]:
        """Get institutional holders.

        Args:
            symbol: Stock symbol.

        Returns:
            Institutional holders data.
        """
        return await self._client.get(f"institutional-holder/{symbol}")

    async def get_mutual_fund_holders(
        self,
        symbol: str,
    ) -> list[dict[str, Any]]:
        """Get mutual fund holders.

        Args:
            symbol: Stock symbol.

        Returns:
            Mutual fund holders data.
        """
        return await self._client.get(f"mutual-fund-holder/{symbol}")

    async def get_etf_holders(
        self,
        symbol: str,
    ) -> list[dict[str, Any]]:
        """Get ETF holders (for an ETF symbol).

        Args:
            symbol: ETF symbol.

        Returns:
            ETF holders data.
        """
        return await self._client.get(f"etf-holder/{symbol}")

    # Social Sentiment
    async def get_social_sentiment(
        self,
        symbol: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get social media sentiment.

        Args:
            symbol: Stock symbol.
            limit: Number of records.

        Returns:
            Social sentiment data.
        """
        return await self._client.get_v4("historical/social-sentiment", params={"symbol": symbol, "limit": limit})

    # Stock Grade
    async def get_stock_grade(
        self,
        symbol: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Get analyst stock grades.

        Args:
            symbol: Stock symbol.
            limit: Number of records.

        Returns:
            Stock grade data.
        """
        return await self._client.get(f"grade/{symbol}", params={"limit": limit})
