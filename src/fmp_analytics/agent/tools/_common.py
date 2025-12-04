"""Common utilities for agent tools.

Shared imports and helper functions used across all tool modules.
"""

import asyncio
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from agno.tools import tool

from fmp_analytics.api.client import FMPClient
from fmp_analytics.metrics.portfolio import PortfolioManagement
from fmp_analytics.pipeline.data_pipeline import DataPipeline
from fmp_analytics.pipeline.analysis_pipeline import AnalysisPipeline


def get_pipeline() -> tuple[DataPipeline, AnalysisPipeline]:
    """Get data and analysis pipelines."""
    client = FMPClient()
    data_pipeline = DataPipeline(client)
    analysis_pipeline = AnalysisPipeline(data_pipeline)
    return data_pipeline, analysis_pipeline


def run_async(coro: Any) -> Any:
    """Run async function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


__all__ = [
    # Standard library
    "asyncio",
    "date",
    "datetime",
    "timedelta",
    "Any",
    # Third-party
    "np",
    "pd",
    "tool",
    # Internal
    "FMPClient",
    "PortfolioManagement",
    "DataPipeline",
    "AnalysisPipeline",
    # Helpers
    "get_pipeline",
    "run_async",
]
