"""Data pipeline for FMP data ingestion and processing.

This module provides a structured data pipeline for:
- Fetching data from FMP API
- Transforming and cleaning data
- Converting to pandas DataFrames
- Caching and storage
"""

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import structlog

from fmp_analytics.api.client import FMPClient
from fmp_analytics.api.endpoints import (
    CalendarAPI,
    CompanyInfoAPI,
    FinancialStatementsAPI,
    HistoricalDataAPI,
    MarketDataAPI,
    StockListAPI,
)
from fmp_analytics.models import (
    BalanceSheet,
    CashFlowStatement,
    CompanyProfile,
    FinancialRatios,
    HistoricalPrice,
    IncomeStatement,
    KeyMetrics,
    Quote,
)

logger = structlog.get_logger(__name__)


class DataPipeline:
    """Data pipeline for FMP data ingestion and processing."""

    def __init__(self, client: FMPClient | None = None):
        """Initialize data pipeline.

        Args:
            client: FMP API client. Creates new one if not provided.
        """
        self._client = client or FMPClient()
        self._stock_list = StockListAPI(self._client)
        self._company_info = CompanyInfoAPI(self._client)
        self._market_data = MarketDataAPI(self._client)
        self._historical = HistoricalDataAPI(self._client)
        self._financials = FinancialStatementsAPI(self._client)
        self._calendar = CalendarAPI(self._client)

    async def __aenter__(self) -> "DataPipeline":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self._client.close()

    # ==================== STOCK DATA ====================

    async def get_stock_universe(
        self,
        exchanges: list[str] | None = None,
        min_market_cap: int | None = None,
        actively_trading: bool = True,
    ) -> pd.DataFrame:
        """Get universe of stocks based on filters.

        Args:
            exchanges: List of exchanges to include.
            min_market_cap: Minimum market cap filter.
            actively_trading: Only include actively trading stocks.

        Returns:
            DataFrame with stock universe.
        """
        logger.info("Fetching stock universe", exchanges=exchanges, min_market_cap=min_market_cap)

        stocks = await self._stock_list.get_stock_list()

        df = pd.DataFrame(stocks)

        if exchanges:
            df = df[df["exchange"].isin(exchanges)]

        if actively_trading:
            df = df[df.get("isActivelyTrading", True) == True]

        if min_market_cap:
            # Need to fetch market caps
            pass

        logger.info("Stock universe loaded", count=len(df))
        return df

    async def get_index_constituents(
        self,
        index: str = "sp500",
    ) -> pd.DataFrame:
        """Get constituents of a major index.

        Args:
            index: Index name (sp500, nasdaq, dowjones).

        Returns:
            DataFrame with index constituents.
        """
        if index.lower() == "sp500":
            data = await self._stock_list.get_sp500_constituents()
        elif index.lower() == "nasdaq":
            data = await self._stock_list.get_nasdaq_constituents()
        elif index.lower() == "dowjones":
            data = await self._stock_list.get_dowjones_constituents()
        else:
            raise ValueError(f"Unknown index: {index}")

        return pd.DataFrame(data)

    # ==================== COMPANY DATA ====================

    async def get_company_profile(
        self,
        symbol: str,
    ) -> CompanyProfile:
        """Get company profile.

        Args:
            symbol: Stock symbol.

        Returns:
            CompanyProfile model.
        """
        data = await self._company_info.get_profile(symbol)
        if data:
            return CompanyProfile(**data[0])
        raise ValueError(f"No profile found for {symbol}")

    async def get_company_profiles_batch(
        self,
        symbols: list[str],
    ) -> list[CompanyProfile]:
        """Get profiles for multiple companies.

        Args:
            symbols: List of stock symbols.

        Returns:
            List of CompanyProfile models.
        """
        data = await self._company_info.get_profiles_batch(symbols)
        return [CompanyProfile(**item) for item in data]

    # ==================== QUOTES ====================

    async def get_quote(
        self,
        symbol: str,
    ) -> Quote:
        """Get current quote for a symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Quote model.
        """
        data = await self._market_data.get_quote(symbol)
        if data:
            return Quote(**data[0])
        raise ValueError(f"No quote found for {symbol}")

    async def get_quotes_batch(
        self,
        symbols: list[str],
    ) -> list[Quote]:
        """Get quotes for multiple symbols.

        Args:
            symbols: List of stock symbols.

        Returns:
            List of Quote models.
        """
        data = await self._market_data.get_quotes_batch(symbols)
        return [Quote(**item) for item in data]

    async def get_quotes_df(
        self,
        symbols: list[str],
    ) -> pd.DataFrame:
        """Get quotes as DataFrame.

        Args:
            symbols: List of stock symbols.

        Returns:
            DataFrame with quotes.
        """
        quotes = await self.get_quotes_batch(symbols)
        return pd.DataFrame([q.model_dump() for q in quotes])

    # ==================== HISTORICAL DATA ====================

    async def get_historical_prices(
        self,
        symbol: str,
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> pd.DataFrame:
        """Get historical daily prices.

        Args:
            symbol: Stock symbol.
            from_date: Start date.
            to_date: End date.

        Returns:
            DataFrame with OHLCV data.
        """
        from_str = str(from_date) if from_date else None
        to_str = str(to_date) if to_date else None

        data = await self._historical.get_historical_price(symbol, from_str, to_str)

        if "historical" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["historical"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["symbol"] = symbol

        return df

    async def get_historical_prices_batch(
        self,
        symbols: list[str],
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> pd.DataFrame:
        """Get historical prices for multiple symbols.

        Args:
            symbols: List of stock symbols.
            from_date: Start date.
            to_date: End date.

        Returns:
            DataFrame with OHLCV data for all symbols.
        """
        dfs = []
        for symbol in symbols:
            try:
                df = await self.get_historical_prices(symbol, from_date, to_date)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.warning("Failed to fetch historical prices", symbol=symbol, error=str(e))

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    async def get_returns(
        self,
        symbol: str,
        from_date: str | date | None = None,
        to_date: str | date | None = None,
        return_type: str = "simple",
    ) -> pd.Series:
        """Get return series for a symbol.

        Args:
            symbol: Stock symbol.
            from_date: Start date.
            to_date: End date.
            return_type: "simple" or "log".

        Returns:
            Series of returns.
        """
        df = await self.get_historical_prices(symbol, from_date, to_date)

        if df.empty:
            return pd.Series()

        if return_type == "log":
            returns = np.log(df["adjClose"] / df["adjClose"].shift(1))
        else:
            returns = df["adjClose"].pct_change()

        returns = returns.dropna()
        returns.index = df.loc[returns.index, "date"]
        returns.name = symbol

        return returns

    async def get_returns_matrix(
        self,
        symbols: list[str],
        from_date: str | date | None = None,
        to_date: str | date | None = None,
        return_type: str = "simple",
    ) -> pd.DataFrame:
        """Get returns matrix for multiple symbols.

        Args:
            symbols: List of stock symbols.
            from_date: Start date.
            to_date: End date.
            return_type: "simple" or "log".

        Returns:
            DataFrame with returns for each symbol.
        """
        returns_dict = {}
        for symbol in symbols:
            try:
                returns = await self.get_returns(symbol, from_date, to_date, return_type)
                if not returns.empty:
                    returns_dict[symbol] = returns
            except Exception as e:
                logger.warning("Failed to fetch returns", symbol=symbol, error=str(e))

        if not returns_dict:
            return pd.DataFrame()

        return pd.DataFrame(returns_dict)

    # ==================== FINANCIAL STATEMENTS ====================

    async def get_income_statements(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get income statements.

        Args:
            symbol: Stock symbol.
            period: "annual" or "quarter".
            limit: Number of periods.

        Returns:
            DataFrame with income statements.
        """
        data = await self._financials.get_income_statement(symbol, period, limit)
        return pd.DataFrame(data)

    async def get_balance_sheets(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get balance sheet statements.

        Args:
            symbol: Stock symbol.
            period: "annual" or "quarter".
            limit: Number of periods.

        Returns:
            DataFrame with balance sheets.
        """
        data = await self._financials.get_balance_sheet(symbol, period, limit)
        return pd.DataFrame(data)

    async def get_cash_flow_statements(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get cash flow statements.

        Args:
            symbol: Stock symbol.
            period: "annual" or "quarter".
            limit: Number of periods.

        Returns:
            DataFrame with cash flow statements.
        """
        data = await self._financials.get_cash_flow_statement(symbol, period, limit)
        return pd.DataFrame(data)

    async def get_financial_ratios(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get financial ratios.

        Args:
            symbol: Stock symbol.
            period: "annual" or "quarter".
            limit: Number of periods.

        Returns:
            DataFrame with financial ratios.
        """
        data = await self._financials.get_ratios(symbol, period, limit)
        return pd.DataFrame(data)

    async def get_key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 10,
    ) -> pd.DataFrame:
        """Get key financial metrics.

        Args:
            symbol: Stock symbol.
            period: "annual" or "quarter".
            limit: Number of periods.

        Returns:
            DataFrame with key metrics.
        """
        data = await self._financials.get_key_metrics(symbol, period, limit)
        return pd.DataFrame(data)

    async def get_dcf_valuation(
        self,
        symbol: str,
    ) -> dict[str, Any]:
        """Get DCF valuation.

        Args:
            symbol: Stock symbol.

        Returns:
            DCF valuation data.
        """
        data = await self._financials.get_dcf(symbol)
        return data[0] if data else {}

    # ==================== CALENDAR DATA ====================

    async def get_earnings_calendar(
        self,
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> pd.DataFrame:
        """Get earnings calendar.

        Args:
            from_date: Start date.
            to_date: End date.

        Returns:
            DataFrame with earnings calendar.
        """
        from_str = str(from_date) if from_date else None
        to_str = str(to_date) if to_date else None

        data = await self._calendar.get_earnings_calendar(from_str, to_str)
        return pd.DataFrame(data)

    async def get_dividend_calendar(
        self,
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> pd.DataFrame:
        """Get dividend calendar.

        Args:
            from_date: Start date.
            to_date: End date.

        Returns:
            DataFrame with dividend calendar.
        """
        from_str = str(from_date) if from_date else None
        to_str = str(to_date) if to_date else None

        data = await self._calendar.get_dividend_calendar(from_str, to_str)
        return pd.DataFrame(data)

    async def get_economic_calendar(
        self,
        from_date: str | date | None = None,
        to_date: str | date | None = None,
    ) -> pd.DataFrame:
        """Get economic calendar.

        Args:
            from_date: Start date.
            to_date: End date.

        Returns:
            DataFrame with economic calendar.
        """
        from_str = str(from_date) if from_date else None
        to_str = str(to_date) if to_date else None

        data = await self._calendar.get_economic_calendar(from_str, to_str)
        return pd.DataFrame(data)

    # ==================== MARKET DATA ====================

    async def get_market_movers(self) -> dict[str, pd.DataFrame]:
        """Get market movers (gainers, losers, most active).

        Returns:
            Dictionary with DataFrames for each category.
        """
        gainers = await self._market_data.get_gainers()
        losers = await self._market_data.get_losers()
        actives = await self._market_data.get_most_active()

        return {
            "gainers": pd.DataFrame(gainers),
            "losers": pd.DataFrame(losers),
            "most_active": pd.DataFrame(actives),
        }

    async def get_sector_performance(self) -> pd.DataFrame:
        """Get sector performance.

        Returns:
            DataFrame with sector performance.
        """
        data = await self._market_data.get_sector_performance()
        return pd.DataFrame(data)

    # ==================== DATA TRANSFORMATIONS ====================

    @staticmethod
    def calculate_returns_from_prices(
        prices: pd.DataFrame,
        price_col: str = "adjClose",
        return_type: str = "simple",
    ) -> pd.DataFrame:
        """Calculate returns from price DataFrame.

        Args:
            prices: DataFrame with prices.
            price_col: Column name for prices.
            return_type: "simple" or "log".

        Returns:
            DataFrame with returns.
        """
        if return_type == "log":
            returns = np.log(prices[price_col] / prices[price_col].shift(1))
        else:
            returns = prices[price_col].pct_change()

        return returns.dropna()

    @staticmethod
    def resample_to_frequency(
        df: pd.DataFrame,
        frequency: str = "M",
        date_col: str = "date",
        price_col: str = "adjClose",
    ) -> pd.DataFrame:
        """Resample price data to different frequency.

        Args:
            df: DataFrame with price data.
            frequency: Target frequency (D, W, M, Q, Y).
            date_col: Date column name.
            price_col: Price column name.

        Returns:
            Resampled DataFrame.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        resampled = df.resample(frequency).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            price_col: "last",
            "volume": "sum",
        })

        return resampled.dropna().reset_index()

    @staticmethod
    def calculate_covariance_matrix(
        returns: pd.DataFrame,
        annualize: bool = True,
        trading_days: int = 252,
    ) -> pd.DataFrame:
        """Calculate covariance matrix from returns.

        Args:
            returns: DataFrame with returns.
            annualize: Whether to annualize.
            trading_days: Trading days per year.

        Returns:
            Covariance matrix.
        """
        cov = returns.cov()
        if annualize:
            cov = cov * trading_days
        return cov

    @staticmethod
    def calculate_correlation_matrix(
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate correlation matrix from returns.

        Args:
            returns: DataFrame with returns.

        Returns:
            Correlation matrix.
        """
        return returns.corr()
