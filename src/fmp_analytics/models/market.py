"""Market data models."""

from datetime import datetime

from pydantic import Field

from fmp_analytics.models.base import FMPBaseModel


class Quote(FMPBaseModel):
    """Stock quote data model."""

    symbol: str = Field(..., description="Stock symbol")
    name: str | None = Field(None, description="Company name")
    price: float | None = Field(None, description="Current price")
    changes_percentage: float | None = Field(None, alias="changesPercentage", description="Change percentage")
    change: float | None = Field(None, description="Price change")
    day_low: float | None = Field(None, alias="dayLow", description="Day low")
    day_high: float | None = Field(None, alias="dayHigh", description="Day high")
    year_high: float | None = Field(None, alias="yearHigh", description="52-week high")
    year_low: float | None = Field(None, alias="yearLow", description="52-week low")
    market_cap: int | None = Field(None, alias="marketCap", description="Market cap")
    price_avg_50: float | None = Field(None, alias="priceAvg50", description="50-day average")
    price_avg_200: float | None = Field(None, alias="priceAvg200", description="200-day average")
    exchange: str | None = Field(None, description="Exchange")
    volume: int | None = Field(None, description="Volume")
    avg_volume: int | None = Field(None, alias="avgVolume", description="Average volume")
    open: float | None = Field(None, description="Open price")
    previous_close: float | None = Field(None, alias="previousClose", description="Previous close")
    eps: float | None = Field(None, description="EPS")
    pe: float | None = Field(None, description="P/E ratio")
    earnings_announcement: str | None = Field(None, alias="earningsAnnouncement")
    shares_outstanding: int | None = Field(None, alias="sharesOutstanding")
    timestamp: int | None = Field(None, description="Unix timestamp")


class ForexQuote(FMPBaseModel):
    """Forex quote data model."""

    ticker: str = Field(..., description="Currency pair")
    bid: float | None = Field(None, description="Bid price")
    ask: float | None = Field(None, description="Ask price")
    open: float | None = Field(None, description="Open price")
    low: float | None = Field(None, description="Low price")
    high: float | None = Field(None, description="High price")
    changes: float | None = Field(None, description="Price change")
    date: str | None = Field(None, description="Date")


class CryptoQuote(FMPBaseModel):
    """Cryptocurrency quote data model."""

    symbol: str = Field(..., description="Crypto symbol")
    name: str | None = Field(None, description="Crypto name")
    price: float | None = Field(None, description="Current price")
    changes_percentage: float | None = Field(None, alias="changesPercentage")
    change: float | None = Field(None, description="Price change")
    day_low: float | None = Field(None, alias="dayLow")
    day_high: float | None = Field(None, alias="dayHigh")
    year_high: float | None = Field(None, alias="yearHigh")
    year_low: float | None = Field(None, alias="yearLow")
    market_cap: int | None = Field(None, alias="marketCap")
    price_avg_50: float | None = Field(None, alias="priceAvg50")
    price_avg_200: float | None = Field(None, alias="priceAvg200")
    volume: int | None = Field(None, description="Volume")
    avg_volume: int | None = Field(None, alias="avgVolume")
    open: float | None = Field(None, description="Open price")
    previous_close: float | None = Field(None, alias="previousClose")
    timestamp: int | None = Field(None, description="Unix timestamp")


class SectorPerformance(FMPBaseModel):
    """Sector performance data model."""

    sector: str = Field(..., description="Sector name")
    changes_percentage: str | None = Field(None, alias="changesPercentage")


class MarketMover(FMPBaseModel):
    """Market mover (gainer/loser/active) data model."""

    symbol: str = Field(..., description="Stock symbol")
    name: str | None = Field(None, description="Company name")
    change: float | None = Field(None, description="Price change")
    price: float | None = Field(None, description="Current price")
    changes_percentage: float | None = Field(None, alias="changesPercentage")


class PrePostMarketQuote(FMPBaseModel):
    """Pre/post market quote data model."""

    symbol: str = Field(..., description="Stock symbol")
    price: float | None = Field(None, description="Current price")
    change: float | None = Field(None, description="Price change")
    timestamp: datetime | None = Field(None, description="Timestamp")


class IndexConstituent(FMPBaseModel):
    """Index constituent data model."""

    symbol: str = Field(..., description="Stock symbol")
    name: str | None = Field(None, description="Company name")
    sector: str | None = Field(None, description="Sector")
    sub_sector: str | None = Field(None, alias="subSector", description="Sub-sector")
    head_quarter: str | None = Field(None, alias="headQuarter", description="Headquarters")
    date_first_added: str | None = Field(None, alias="dateFirstAdded")
    cik: str | None = Field(None, description="CIK number")
    founded: str | None = Field(None, description="Founded year")
