"""Financial metrics calculation modules.

This module provides comprehensive financial metrics calculations based on
CFA (Chartered Financial Analyst), FRM (Financial Risk Manager),
CQF (Certificate in Quantitative Finance), and Portfolio Management curriculum.

Based on Norte Asset Management Quant Finance Master Guide with 150+ models.
"""

from fmp_analytics.metrics.cfa import CFAMetrics
from fmp_analytics.metrics.frm import FRMMetrics
from fmp_analytics.metrics.cqf import CQFMetrics
from fmp_analytics.metrics.portfolio import PortfolioManagement

__all__ = [
    "CFAMetrics",
    "FRMMetrics",
    "CQFMetrics",
    "PortfolioManagement",
]
