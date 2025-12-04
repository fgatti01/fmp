"""Base class for FMP API endpoint modules."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fmp_analytics.api.client import FMPClient


class BaseAPI:
    """Base class for API endpoint modules."""

    def __init__(self, client: "FMPClient"):
        """Initialize API module.

        Args:
            client: FMP API client instance.
        """
        self._client = client
