"""Portfolio and holding data models."""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from uuid import UUID, uuid4

from pydantic import Field

from fmp_analytics.models.base import FMPBaseModel


class TransactionType(str, Enum):
    """Transaction type enumeration."""

    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    SPLIT = "split"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"


class AssetClass(str, Enum):
    """Asset class enumeration."""

    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    REAL_ESTATE = "real_estate"
    ALTERNATIVE = "alternative"
    CASH = "cash"


class Transaction(FMPBaseModel):
    """Transaction data model."""

    id: UUID = Field(default_factory=uuid4, description="Transaction ID")
    symbol: str = Field(..., description="Stock symbol")
    transaction_type: TransactionType = Field(..., alias="transactionType", description="Transaction type")
    quantity: Decimal = Field(..., description="Number of shares")
    price: Decimal = Field(..., description="Price per share")
    total_amount: Decimal = Field(..., alias="totalAmount", description="Total transaction amount")
    commission: Decimal = Field(default=Decimal("0"), description="Commission/fees")
    transaction_date: date = Field(..., alias="transactionDate", description="Transaction date")
    notes: str | None = Field(None, description="Transaction notes")
    created_at: datetime = Field(default_factory=datetime.now, alias="createdAt")


class Holding(FMPBaseModel):
    """Portfolio holding data model."""

    symbol: str = Field(..., description="Stock symbol")
    name: str | None = Field(None, description="Company name")
    quantity: Decimal = Field(..., description="Number of shares")
    average_cost: Decimal = Field(..., alias="averageCost", description="Average cost per share")
    current_price: Decimal | None = Field(None, alias="currentPrice", description="Current market price")
    market_value: Decimal | None = Field(None, alias="marketValue", description="Current market value")
    cost_basis: Decimal | None = Field(None, alias="costBasis", description="Total cost basis")
    unrealized_gain: Decimal | None = Field(None, alias="unrealizedGain", description="Unrealized gain/loss")
    unrealized_gain_percent: Decimal | None = Field(None, alias="unrealizedGainPercent", description="Unrealized gain %")
    realized_gain: Decimal = Field(default=Decimal("0"), alias="realizedGain", description="Realized gain/loss")
    asset_class: AssetClass = Field(default=AssetClass.EQUITY, alias="assetClass", description="Asset class")
    sector: str | None = Field(None, description="Sector")
    weight: Decimal | None = Field(None, description="Portfolio weight %")
    last_updated: datetime = Field(default_factory=datetime.now, alias="lastUpdated")

    def update_market_value(self, current_price: Decimal) -> None:
        """Update holding with current market price.

        Args:
            current_price: Current market price.
        """
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        self.cost_basis = self.quantity * self.average_cost
        self.unrealized_gain = self.market_value - self.cost_basis
        if self.cost_basis > 0:
            self.unrealized_gain_percent = (self.unrealized_gain / self.cost_basis) * 100
        self.last_updated = datetime.now()


class PortfolioMetrics(FMPBaseModel):
    """Portfolio-level metrics data model."""

    total_value: Decimal = Field(..., alias="totalValue", description="Total portfolio value")
    total_cost: Decimal = Field(..., alias="totalCost", description="Total cost basis")
    total_unrealized_gain: Decimal = Field(..., alias="totalUnrealizedGain", description="Total unrealized gain")
    total_unrealized_gain_percent: Decimal = Field(..., alias="totalUnrealizedGainPercent")
    total_realized_gain: Decimal = Field(default=Decimal("0"), alias="totalRealizedGain")
    cash_balance: Decimal = Field(default=Decimal("0"), alias="cashBalance", description="Cash balance")
    num_holdings: int = Field(..., alias="numHoldings", description="Number of holdings")

    # Risk Metrics
    beta: float | None = Field(None, description="Portfolio beta")
    sharpe_ratio: float | None = Field(None, alias="sharpeRatio", description="Sharpe ratio")
    sortino_ratio: float | None = Field(None, alias="sortinoRatio", description="Sortino ratio")
    max_drawdown: float | None = Field(None, alias="maxDrawdown", description="Maximum drawdown")
    volatility: float | None = Field(None, description="Portfolio volatility (annualized)")
    var_95: float | None = Field(None, alias="var95", description="Value at Risk (95%)")
    cvar_95: float | None = Field(None, alias="cvar95", description="Conditional VaR (95%)")

    # Return Metrics
    daily_return: float | None = Field(None, alias="dailyReturn", description="Daily return")
    weekly_return: float | None = Field(None, alias="weeklyReturn", description="Weekly return")
    monthly_return: float | None = Field(None, alias="monthlyReturn", description="Monthly return")
    ytd_return: float | None = Field(None, alias="ytdReturn", description="YTD return")
    annualized_return: float | None = Field(None, alias="annualizedReturn", description="Annualized return")

    # Diversification
    concentration_ratio: float | None = Field(None, alias="concentrationRatio", description="Concentration (Herfindahl)")
    num_sectors: int | None = Field(None, alias="numSectors", description="Number of sectors")
    num_asset_classes: int | None = Field(None, alias="numAssetClasses", description="Number of asset classes")

    as_of_date: datetime = Field(default_factory=datetime.now, alias="asOfDate")


class Portfolio(FMPBaseModel):
    """Portfolio data model."""

    id: UUID = Field(default_factory=uuid4, description="Portfolio ID")
    name: str = Field(..., description="Portfolio name")
    description: str | None = Field(None, description="Portfolio description")
    currency: str = Field(default="USD", description="Base currency")
    holdings: list[Holding] = Field(default_factory=list, description="Portfolio holdings")
    transactions: list[Transaction] = Field(default_factory=list, description="Transaction history")
    metrics: PortfolioMetrics | None = Field(None, description="Portfolio metrics")
    benchmark_symbol: str = Field(default="SPY", alias="benchmarkSymbol", description="Benchmark symbol")
    created_at: datetime = Field(default_factory=datetime.now, alias="createdAt")
    updated_at: datetime = Field(default_factory=datetime.now, alias="updatedAt")

    def get_holding(self, symbol: str) -> Holding | None:
        """Get holding by symbol.

        Args:
            symbol: Stock symbol.

        Returns:
            Holding if found, None otherwise.
        """
        for holding in self.holdings:
            if holding.symbol.upper() == symbol.upper():
                return holding
        return None

    def get_total_value(self) -> Decimal:
        """Calculate total portfolio value.

        Returns:
            Total portfolio value.
        """
        return sum(
            (h.market_value or Decimal("0")) for h in self.holdings
        )

    def get_allocation(self) -> dict[str, Decimal]:
        """Get portfolio allocation by symbol.

        Returns:
            Dictionary of symbol to weight percentage.
        """
        total = self.get_total_value()
        if total == 0:
            return {}
        return {
            h.symbol: ((h.market_value or Decimal("0")) / total) * 100
            for h in self.holdings
        }

    def get_sector_allocation(self) -> dict[str, Decimal]:
        """Get portfolio allocation by sector.

        Returns:
            Dictionary of sector to weight percentage.
        """
        total = self.get_total_value()
        if total == 0:
            return {}

        sector_values: dict[str, Decimal] = {}
        for holding in self.holdings:
            sector = holding.sector or "Unknown"
            sector_values[sector] = sector_values.get(sector, Decimal("0")) + (
                holding.market_value or Decimal("0")
            )

        return {
            sector: (value / total) * 100
            for sector, value in sector_values.items()
        }
