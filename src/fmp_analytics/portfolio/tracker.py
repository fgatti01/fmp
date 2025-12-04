"""Portfolio tracker for real-time portfolio monitoring.

This module provides functionality to:
- Track portfolio holdings and transactions
- Update market values in real-time
- Calculate portfolio performance metrics
- Monitor risk levels
"""

import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import structlog

from fmp_analytics.models.portfolio import (
    AssetClass,
    Holding,
    Portfolio,
    PortfolioMetrics,
    Transaction,
    TransactionType,
)
from fmp_analytics.pipeline.data_pipeline import DataPipeline

logger = structlog.get_logger(__name__)


class PortfolioTracker:
    """Real-time portfolio tracker."""

    def __init__(
        self,
        portfolio: Portfolio | None = None,
        data_pipeline: DataPipeline | None = None,
        storage_path: Path | None = None,
    ):
        """Initialize portfolio tracker.

        Args:
            portfolio: Existing portfolio or None to create new.
            data_pipeline: Data pipeline for market data.
            storage_path: Path to persist portfolio data.
        """
        self.portfolio = portfolio or Portfolio(name="My Portfolio")
        self._data = data_pipeline or DataPipeline()
        self._storage_path = storage_path

        if self._storage_path and self._storage_path.exists():
            self._load_portfolio()

    def _load_portfolio(self) -> None:
        """Load portfolio from storage."""
        if self._storage_path and self._storage_path.exists():
            try:
                data = json.loads(self._storage_path.read_text())
                self.portfolio = Portfolio(**data)
                logger.info("Portfolio loaded", name=self.portfolio.name)
            except Exception as e:
                logger.error("Failed to load portfolio", error=str(e))

    def _save_portfolio(self) -> None:
        """Save portfolio to storage."""
        if self._storage_path:
            try:
                self._storage_path.write_text(
                    self.portfolio.model_dump_json(indent=2)
                )
                logger.info("Portfolio saved", name=self.portfolio.name)
            except Exception as e:
                logger.error("Failed to save portfolio", error=str(e))

    def add_transaction(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: float,
        price: float,
        commission: float = 0.0,
        transaction_date: datetime | None = None,
        notes: str | None = None,
    ) -> Transaction:
        """Add a transaction to the portfolio.

        Args:
            symbol: Stock symbol.
            transaction_type: Type of transaction.
            quantity: Number of shares.
            price: Price per share.
            commission: Commission/fees.
            transaction_date: Date of transaction.
            notes: Optional notes.

        Returns:
            Created Transaction.
        """
        symbol = symbol.upper()
        quantity_dec = Decimal(str(quantity))
        price_dec = Decimal(str(price))
        commission_dec = Decimal(str(commission))

        transaction = Transaction(
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity_dec,
            price=price_dec,
            total_amount=quantity_dec * price_dec + commission_dec,
            commission=commission_dec,
            transaction_date=transaction_date or datetime.now().date(),
            notes=notes,
        )

        self.portfolio.transactions.append(transaction)

        # Update holdings
        self._update_holdings_from_transaction(transaction)

        self.portfolio.updated_at = datetime.now()
        self._save_portfolio()

        logger.info(
            "Transaction added",
            symbol=symbol,
            type=transaction_type,
            quantity=quantity,
            price=price,
        )

        return transaction

    def _update_holdings_from_transaction(self, transaction: Transaction) -> None:
        """Update holdings based on a transaction.

        Args:
            transaction: Transaction to process.
        """
        holding = self.portfolio.get_holding(transaction.symbol)

        if transaction.transaction_type == TransactionType.BUY:
            if holding:
                # Update average cost and quantity
                old_value = holding.quantity * holding.average_cost
                new_value = transaction.quantity * transaction.price
                new_quantity = holding.quantity + transaction.quantity
                holding.quantity = new_quantity
                holding.average_cost = (old_value + new_value) / new_quantity
            else:
                # Create new holding
                holding = Holding(
                    symbol=transaction.symbol,
                    quantity=transaction.quantity,
                    average_cost=transaction.price,
                )
                self.portfolio.holdings.append(holding)

        elif transaction.transaction_type == TransactionType.SELL:
            if holding:
                # Calculate realized gain
                cost_basis = holding.average_cost * transaction.quantity
                proceeds = transaction.quantity * transaction.price
                realized_gain = proceeds - cost_basis
                holding.realized_gain += realized_gain
                holding.quantity -= transaction.quantity

                # Remove holding if fully sold
                if holding.quantity <= 0:
                    self.portfolio.holdings = [
                        h for h in self.portfolio.holdings
                        if h.symbol != transaction.symbol
                    ]

        elif transaction.transaction_type == TransactionType.DIVIDEND:
            if holding:
                holding.realized_gain += transaction.total_amount

    def buy(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        notes: str | None = None,
    ) -> Transaction:
        """Execute a buy transaction.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares.
            price: Price per share.
            commission: Commission/fees.
            notes: Optional notes.

        Returns:
            Created Transaction.
        """
        return self.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.BUY,
            quantity=quantity,
            price=price,
            commission=commission,
            notes=notes,
        )

    def sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        commission: float = 0.0,
        notes: str | None = None,
    ) -> Transaction:
        """Execute a sell transaction.

        Args:
            symbol: Stock symbol.
            quantity: Number of shares.
            price: Price per share.
            commission: Commission/fees.
            notes: Optional notes.

        Returns:
            Created Transaction.
        """
        return self.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.SELL,
            quantity=quantity,
            price=price,
            commission=commission,
            notes=notes,
        )

    async def update_market_values(self) -> None:
        """Update all holdings with current market values."""
        if not self.portfolio.holdings:
            return

        symbols = [h.symbol for h in self.portfolio.holdings]

        try:
            quotes = await self._data.get_quotes_batch(symbols)
            profiles = await self._data.get_company_profiles_batch(symbols)

            # Create lookup dicts
            quote_lookup = {q.symbol: q for q in quotes}
            profile_lookup = {p.symbol: p for p in profiles}

            for holding in self.portfolio.holdings:
                quote = quote_lookup.get(holding.symbol)
                profile = profile_lookup.get(holding.symbol)

                if quote and quote.price:
                    holding.update_market_value(Decimal(str(quote.price)))
                    holding.name = quote.name

                if profile:
                    holding.sector = profile.sector

            # Calculate weights
            total_value = self.portfolio.get_total_value()
            for holding in self.portfolio.holdings:
                if total_value > 0 and holding.market_value:
                    holding.weight = (holding.market_value / total_value) * 100

            self.portfolio.updated_at = datetime.now()
            self._save_portfolio()

            logger.info("Market values updated", holdings=len(self.portfolio.holdings))

        except Exception as e:
            logger.error("Failed to update market values", error=str(e))
            raise

    async def calculate_metrics(
        self,
        benchmark_symbol: str = "SPY",
        risk_free_rate: float = 0.05,
    ) -> PortfolioMetrics:
        """Calculate portfolio metrics.

        Args:
            benchmark_symbol: Benchmark symbol for relative metrics.
            risk_free_rate: Risk-free rate for Sharpe ratio.

        Returns:
            PortfolioMetrics with calculated values.
        """
        await self.update_market_values()

        total_value = self.portfolio.get_total_value()
        total_cost = sum(
            h.cost_basis or Decimal("0") for h in self.portfolio.holdings
        )
        total_unrealized = total_value - total_cost
        total_unrealized_pct = (
            (total_unrealized / total_cost) * 100 if total_cost > 0 else Decimal("0")
        )
        total_realized = sum(h.realized_gain for h in self.portfolio.holdings)

        metrics = PortfolioMetrics(
            total_value=total_value,
            total_cost=total_cost,
            total_unrealized_gain=total_unrealized,
            total_unrealized_gain_percent=total_unrealized_pct,
            total_realized_gain=total_realized,
            num_holdings=len(self.portfolio.holdings),
            num_sectors=len(set(h.sector for h in self.portfolio.holdings if h.sector)),
        )

        self.portfolio.metrics = metrics
        self._save_portfolio()

        return metrics

    def get_holdings_summary(self) -> list[dict[str, Any]]:
        """Get summary of all holdings.

        Returns:
            List of holding summaries.
        """
        return [
            {
                "symbol": h.symbol,
                "name": h.name,
                "quantity": float(h.quantity),
                "average_cost": float(h.average_cost),
                "current_price": float(h.current_price) if h.current_price else None,
                "market_value": float(h.market_value) if h.market_value else None,
                "unrealized_gain": float(h.unrealized_gain) if h.unrealized_gain else None,
                "unrealized_gain_pct": float(h.unrealized_gain_percent) if h.unrealized_gain_percent else None,
                "weight": float(h.weight) if h.weight else None,
                "sector": h.sector,
            }
            for h in self.portfolio.holdings
        ]

    def get_allocation(self) -> dict[str, float]:
        """Get portfolio allocation by symbol.

        Returns:
            Dictionary of symbol to weight percentage.
        """
        return {
            sym: float(weight)
            for sym, weight in self.portfolio.get_allocation().items()
        }

    def get_sector_allocation(self) -> dict[str, float]:
        """Get portfolio allocation by sector.

        Returns:
            Dictionary of sector to weight percentage.
        """
        return {
            sector: float(weight)
            for sector, weight in self.portfolio.get_sector_allocation().items()
        }

    def get_transactions_history(
        self,
        symbol: str | None = None,
        transaction_type: TransactionType | None = None,
        limit: int = 100,
    ) -> list[Transaction]:
        """Get transaction history.

        Args:
            symbol: Filter by symbol.
            transaction_type: Filter by transaction type.
            limit: Maximum transactions to return.

        Returns:
            List of transactions.
        """
        transactions = self.portfolio.transactions

        if symbol:
            transactions = [t for t in transactions if t.symbol == symbol.upper()]

        if transaction_type:
            transactions = [t for t in transactions if t.transaction_type == transaction_type]

        # Sort by date descending
        transactions = sorted(
            transactions,
            key=lambda t: t.transaction_date,
            reverse=True,
        )

        return transactions[:limit]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get portfolio performance summary.

        Returns:
            Performance summary dictionary.
        """
        metrics = self.portfolio.metrics

        return {
            "total_value": float(metrics.total_value) if metrics else 0,
            "total_cost": float(metrics.total_cost) if metrics else 0,
            "total_gain": float(metrics.total_unrealized_gain) if metrics else 0,
            "total_gain_pct": float(metrics.total_unrealized_gain_percent) if metrics else 0,
            "realized_gain": float(metrics.total_realized_gain) if metrics else 0,
            "num_holdings": metrics.num_holdings if metrics else 0,
            "sharpe_ratio": metrics.sharpe_ratio if metrics else None,
            "beta": metrics.beta if metrics else None,
            "volatility": metrics.volatility if metrics else None,
            "max_drawdown": metrics.max_drawdown if metrics else None,
        }
