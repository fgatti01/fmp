"""Historical price data models."""

from datetime import date, datetime

from pydantic import Field

from fmp_analytics.models.base import FMPBaseModel


class HistoricalPrice(FMPBaseModel):
    """Historical daily price data model."""

    date: date = Field(..., description="Date")
    open: float | None = Field(None, description="Open price")
    high: float | None = Field(None, description="High price")
    low: float | None = Field(None, description="Low price")
    close: float | None = Field(None, description="Close price")
    adj_close: float | None = Field(None, alias="adjClose", description="Adjusted close")
    volume: int | None = Field(None, description="Volume")
    unadjusted_volume: int | None = Field(None, alias="unadjustedVolume")
    change: float | None = Field(None, description="Price change")
    change_percent: float | None = Field(None, alias="changePercent")
    vwap: float | None = Field(None, description="VWAP")
    label: str | None = Field(None, description="Date label")
    change_over_time: float | None = Field(None, alias="changeOverTime")


class IntradayPrice(FMPBaseModel):
    """Intraday price data model."""

    date: datetime = Field(..., description="Timestamp")
    open: float | None = Field(None, description="Open price")
    high: float | None = Field(None, description="High price")
    low: float | None = Field(None, description="Low price")
    close: float | None = Field(None, description="Close price")
    volume: int | None = Field(None, description="Volume")


class DividendHistory(FMPBaseModel):
    """Dividend history data model."""

    date: date = Field(..., description="Date")
    label: str | None = Field(None, description="Date label")
    adj_dividend: float | None = Field(None, alias="adjDividend", description="Adjusted dividend")
    dividend: float | None = Field(None, description="Dividend amount")
    record_date: date | None = Field(None, alias="recordDate")
    payment_date: date | None = Field(None, alias="paymentDate")
    declaration_date: date | None = Field(None, alias="declarationDate")


class StockSplit(FMPBaseModel):
    """Stock split data model."""

    date: date = Field(..., description="Date")
    label: str | None = Field(None, description="Date label")
    numerator: float | None = Field(None, description="Split numerator")
    denominator: float | None = Field(None, description="Split denominator")


class TechnicalIndicator(FMPBaseModel):
    """Technical indicator data model."""

    date: datetime = Field(..., description="Timestamp")
    open: float | None = Field(None, description="Open price")
    high: float | None = Field(None, description="High price")
    low: float | None = Field(None, description="Low price")
    close: float | None = Field(None, description="Close price")
    volume: int | None = Field(None, description="Volume")

    # Indicators (populated based on request)
    sma: float | None = Field(None, description="Simple Moving Average")
    ema: float | None = Field(None, description="Exponential Moving Average")
    wma: float | None = Field(None, description="Weighted Moving Average")
    dema: float | None = Field(None, description="Double EMA")
    tema: float | None = Field(None, description="Triple EMA")
    williams: float | None = Field(None, description="Williams %R")
    rsi: float | None = Field(None, description="RSI")
    adx: float | None = Field(None, description="ADX")
    standard_deviation: float | None = Field(None, alias="standardDeviation")


class HistoricalPriceResponse(FMPBaseModel):
    """Response wrapper for historical price data."""

    symbol: str = Field(..., description="Stock symbol")
    historical: list[HistoricalPrice] = Field(default_factory=list, description="Historical prices")


class DividendHistoryResponse(FMPBaseModel):
    """Response wrapper for dividend history data."""

    symbol: str = Field(..., description="Stock symbol")
    historical: list[DividendHistory] = Field(default_factory=list, description="Dividend history")


class StockSplitHistoryResponse(FMPBaseModel):
    """Response wrapper for stock split history data."""

    symbol: str = Field(..., description="Stock symbol")
    historical: list[StockSplit] = Field(default_factory=list, description="Split history")
