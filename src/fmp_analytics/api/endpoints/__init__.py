"""FMP API endpoint modules organized by category."""

from fmp_analytics.api.endpoints.calendar import CalendarAPI
from fmp_analytics.api.endpoints.company import CompanyInfoAPI
from fmp_analytics.api.endpoints.financials import FinancialStatementsAPI
from fmp_analytics.api.endpoints.historical import HistoricalDataAPI
from fmp_analytics.api.endpoints.market import MarketDataAPI
from fmp_analytics.api.endpoints.stock_list import StockListAPI

__all__ = [
    "StockListAPI",
    "CompanyInfoAPI",
    "MarketDataAPI",
    "HistoricalDataAPI",
    "FinancialStatementsAPI",
    "CalendarAPI",
]
