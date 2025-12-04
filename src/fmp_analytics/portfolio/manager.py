"""Portfolio manager for multiple portfolios.

This module provides functionality to:
- Manage multiple portfolios
- Aggregate portfolio analytics
- Generate reports
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog

from fmp_analytics.models.portfolio import Portfolio
from fmp_analytics.pipeline.data_pipeline import DataPipeline
from fmp_analytics.pipeline.analysis_pipeline import AnalysisPipeline
from fmp_analytics.portfolio.tracker import PortfolioTracker

logger = structlog.get_logger(__name__)


class PortfolioManager:
    """Manager for multiple portfolios."""

    def __init__(
        self,
        storage_dir: Path | None = None,
        data_pipeline: DataPipeline | None = None,
    ):
        """Initialize portfolio manager.

        Args:
            storage_dir: Directory to store portfolio data.
            data_pipeline: Data pipeline for market data.
        """
        self._storage_dir = storage_dir or Path.home() / ".fmp_analytics" / "portfolios"
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._data = data_pipeline or DataPipeline()
        self._analysis = AnalysisPipeline(self._data)
        self._trackers: dict[UUID, PortfolioTracker] = {}

        self._load_portfolios()

    def _load_portfolios(self) -> None:
        """Load all portfolios from storage."""
        for file_path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text())
                portfolio = Portfolio(**data)
                tracker = PortfolioTracker(
                    portfolio=portfolio,
                    data_pipeline=self._data,
                    storage_path=file_path,
                )
                self._trackers[portfolio.id] = tracker
                logger.info("Portfolio loaded", name=portfolio.name, id=str(portfolio.id))
            except Exception as e:
                logger.error("Failed to load portfolio", file=str(file_path), error=str(e))

    def create_portfolio(
        self,
        name: str,
        description: str | None = None,
        currency: str = "USD",
        benchmark_symbol: str = "SPY",
    ) -> PortfolioTracker:
        """Create a new portfolio.

        Args:
            name: Portfolio name.
            description: Portfolio description.
            currency: Base currency.
            benchmark_symbol: Benchmark symbol.

        Returns:
            PortfolioTracker for the new portfolio.
        """
        portfolio = Portfolio(
            name=name,
            description=description,
            currency=currency,
            benchmark_symbol=benchmark_symbol,
        )

        storage_path = self._storage_dir / f"{portfolio.id}.json"

        tracker = PortfolioTracker(
            portfolio=portfolio,
            data_pipeline=self._data,
            storage_path=storage_path,
        )

        tracker._save_portfolio()
        self._trackers[portfolio.id] = tracker

        logger.info("Portfolio created", name=name, id=str(portfolio.id))
        return tracker

    def get_portfolio(self, portfolio_id: UUID | str) -> PortfolioTracker | None:
        """Get a portfolio tracker by ID.

        Args:
            portfolio_id: Portfolio UUID.

        Returns:
            PortfolioTracker if found, None otherwise.
        """
        if isinstance(portfolio_id, str):
            portfolio_id = UUID(portfolio_id)
        return self._trackers.get(portfolio_id)

    def get_portfolio_by_name(self, name: str) -> PortfolioTracker | None:
        """Get a portfolio tracker by name.

        Args:
            name: Portfolio name.

        Returns:
            PortfolioTracker if found, None otherwise.
        """
        for tracker in self._trackers.values():
            if tracker.portfolio.name.lower() == name.lower():
                return tracker
        return None

    def list_portfolios(self) -> list[dict[str, Any]]:
        """List all portfolios.

        Returns:
            List of portfolio summaries.
        """
        return [
            {
                "id": str(t.portfolio.id),
                "name": t.portfolio.name,
                "description": t.portfolio.description,
                "currency": t.portfolio.currency,
                "num_holdings": len(t.portfolio.holdings),
                "created_at": t.portfolio.created_at.isoformat(),
                "updated_at": t.portfolio.updated_at.isoformat(),
            }
            for t in self._trackers.values()
        ]

    def delete_portfolio(self, portfolio_id: UUID | str) -> bool:
        """Delete a portfolio.

        Args:
            portfolio_id: Portfolio UUID.

        Returns:
            True if deleted, False if not found.
        """
        if isinstance(portfolio_id, str):
            portfolio_id = UUID(portfolio_id)

        tracker = self._trackers.get(portfolio_id)
        if not tracker:
            return False

        # Delete storage file
        storage_path = self._storage_dir / f"{portfolio_id}.json"
        if storage_path.exists():
            storage_path.unlink()

        del self._trackers[portfolio_id]
        logger.info("Portfolio deleted", id=str(portfolio_id))
        return True

    async def update_all_portfolios(self) -> None:
        """Update market values for all portfolios."""
        for tracker in self._trackers.values():
            try:
                await tracker.update_market_values()
            except Exception as e:
                logger.error(
                    "Failed to update portfolio",
                    name=tracker.portfolio.name,
                    error=str(e),
                )

    async def get_aggregate_metrics(self) -> dict[str, Any]:
        """Get aggregate metrics across all portfolios.

        Returns:
            Aggregate metrics dictionary.
        """
        await self.update_all_portfolios()

        total_value = sum(
            float(t.portfolio.get_total_value())
            for t in self._trackers.values()
        )

        total_holdings = sum(
            len(t.portfolio.holdings)
            for t in self._trackers.values()
        )

        # Aggregate holdings across portfolios
        all_holdings: dict[str, float] = {}
        for tracker in self._trackers.values():
            for holding in tracker.portfolio.holdings:
                symbol = holding.symbol
                value = float(holding.market_value or 0)
                all_holdings[symbol] = all_holdings.get(symbol, 0) + value

        # Top holdings
        top_holdings = sorted(
            all_holdings.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]

        return {
            "num_portfolios": len(self._trackers),
            "total_value": total_value,
            "total_holdings": total_holdings,
            "unique_symbols": len(all_holdings),
            "top_holdings": [
                {"symbol": symbol, "value": value, "weight": value / total_value * 100}
                for symbol, value in top_holdings
            ] if total_value > 0 else [],
        }

    async def generate_report(
        self,
        portfolio_id: UUID | str,
    ) -> dict[str, Any]:
        """Generate comprehensive portfolio report.

        Args:
            portfolio_id: Portfolio UUID.

        Returns:
            Portfolio report dictionary.
        """
        tracker = self.get_portfolio(portfolio_id)
        if not tracker:
            raise ValueError(f"Portfolio not found: {portfolio_id}")

        await tracker.update_market_values()
        metrics = await tracker.calculate_metrics()

        # Get holdings data
        holdings = tracker.get_holdings_summary()

        # Get allocations
        symbol_allocation = tracker.get_allocation()
        sector_allocation = tracker.get_sector_allocation()

        # Get recent transactions
        recent_transactions = tracker.get_transactions_history(limit=10)

        return {
            "portfolio": {
                "id": str(tracker.portfolio.id),
                "name": tracker.portfolio.name,
                "description": tracker.portfolio.description,
                "currency": tracker.portfolio.currency,
                "benchmark": tracker.portfolio.benchmark_symbol,
            },
            "summary": {
                "total_value": float(metrics.total_value),
                "total_cost": float(metrics.total_cost),
                "unrealized_gain": float(metrics.total_unrealized_gain),
                "unrealized_gain_pct": float(metrics.total_unrealized_gain_percent),
                "realized_gain": float(metrics.total_realized_gain),
                "num_holdings": metrics.num_holdings,
            },
            "holdings": holdings,
            "allocation": {
                "by_symbol": symbol_allocation,
                "by_sector": sector_allocation,
            },
            "risk_metrics": {
                "beta": metrics.beta,
                "volatility": metrics.volatility,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown": metrics.max_drawdown,
                "var_95": metrics.var_95,
            },
            "recent_transactions": [
                {
                    "date": t.transaction_date.isoformat(),
                    "symbol": t.symbol,
                    "type": t.transaction_type.value,
                    "quantity": float(t.quantity),
                    "price": float(t.price),
                    "total": float(t.total_amount),
                }
                for t in recent_transactions
            ],
            "generated_at": datetime.now().isoformat(),
        }

    async def compare_portfolios(
        self,
        portfolio_ids: list[UUID | str],
    ) -> dict[str, Any]:
        """Compare multiple portfolios.

        Args:
            portfolio_ids: List of portfolio UUIDs.

        Returns:
            Comparison data dictionary.
        """
        comparisons = []

        for pid in portfolio_ids:
            tracker = self.get_portfolio(pid)
            if tracker:
                await tracker.update_market_values()
                metrics = await tracker.calculate_metrics()

                comparisons.append({
                    "id": str(tracker.portfolio.id),
                    "name": tracker.portfolio.name,
                    "total_value": float(metrics.total_value),
                    "return": float(metrics.total_unrealized_gain_percent),
                    "num_holdings": metrics.num_holdings,
                    "sharpe_ratio": metrics.sharpe_ratio,
                    "volatility": metrics.volatility,
                    "beta": metrics.beta,
                    "max_drawdown": metrics.max_drawdown,
                })

        return {
            "portfolios": comparisons,
            "compared_at": datetime.now().isoformat(),
        }
