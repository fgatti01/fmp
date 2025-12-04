"""Core FMP API client with rate limiting and error handling."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import httpx
import structlog

from fmp_analytics.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, max_requests: int, period: int):
        """Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the period.
            period: Time period in seconds.
        """
        self.max_requests = max_requests
        self.period = period
        self.tokens = max_requests
        self.last_update = datetime.now()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = datetime.now()
            elapsed = (now - self.last_update).total_seconds()
            self.tokens = min(
                self.max_requests,
                self.tokens + (elapsed * self.max_requests / self.period),
            )
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * self.period / self.max_requests
                logger.debug("Rate limit reached, waiting", wait_time=wait_time)
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


class FMPError(Exception):
    """Base exception for FMP API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        """Initialize FMP error.

        Args:
            message: Error message.
            status_code: HTTP status code if applicable.
        """
        super().__init__(message)
        self.status_code = status_code


class FMPRateLimitError(FMPError):
    """Rate limit exceeded error."""

    pass


class FMPAuthError(FMPError):
    """Authentication error."""

    pass


class FMPNotFoundError(FMPError):
    """Resource not found error."""

    pass


class FMPClient:
    """Async HTTP client for FMP API with rate limiting."""

    def __init__(self, settings: Settings | None = None):
        """Initialize FMP client.

        Args:
            settings: Application settings. Uses default if not provided.
        """
        self.settings = settings or get_settings()
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = RateLimiter(
            max_requests=self.settings.rate_limit_requests,
            period=self.settings.rate_limit_period,
        )

    @asynccontextmanager
    async def _get_client(self) -> AsyncIterator[httpx.AsyncClient]:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
        try:
            yield self._client
        except Exception:
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "FMPClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _build_url(self, endpoint: str, version: str = "stable") -> str:
        """Build full URL for an endpoint.

        Args:
            endpoint: API endpoint path.
            version: API version (stable, v3, or v4).

        Returns:
            Full URL string.
        """
        base = self.settings.fmp_base_url
        if version == "stable":
            return f"{base}/stable/{endpoint.lstrip('/')}"
        return f"{base}/api/{version}/{endpoint.lstrip('/')}"

    async def _request(
        self,
        method: str,
        endpoint: str,
        version: str = "stable",
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Make an API request with rate limiting and error handling.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            version: API version (stable, v3, or v4).
            params: Query parameters.
            **kwargs: Additional request arguments.

        Returns:
            Parsed JSON response.

        Raises:
            FMPError: On API errors.
        """
        await self._rate_limiter.acquire()

        url = self._build_url(endpoint, version)
        params = params or {}
        params["apikey"] = self.settings.fmp_api_key

        logger.debug("Making API request", method=method, url=url, params=params)

        async with self._get_client() as client:
            response = await client.request(method, url, params=params, **kwargs)

            if response.status_code == 401:
                raise FMPAuthError("Invalid API key", status_code=401)
            elif response.status_code == 403:
                raise FMPAuthError("Access forbidden - check API plan", status_code=403)
            elif response.status_code == 404:
                raise FMPNotFoundError("Resource not found", status_code=404)
            elif response.status_code == 429:
                raise FMPRateLimitError("Rate limit exceeded", status_code=429)
            elif response.status_code >= 400:
                raise FMPError(
                    f"API error: {response.text}",
                    status_code=response.status_code,
                )

            return response.json()

    async def get(
        self,
        endpoint: str,
        version: str = "stable",
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make a GET request.

        Args:
            endpoint: API endpoint.
            version: API version (stable, v3, or v4).
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        return await self._request("GET", endpoint, version, params)

    async def get_stable(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make a GET request to stable API (recommended).

        Args:
            endpoint: API endpoint.
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        return await self._request("GET", endpoint, "stable", params)

    async def get_v3(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make a GET request to legacy v3 API.

        Args:
            endpoint: API endpoint.
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        return await self._request("GET", endpoint, "v3", params)

    async def get_v4(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Make a GET request to legacy v4 API.

        Args:
            endpoint: API endpoint.
            params: Query parameters.

        Returns:
            Parsed JSON response.
        """
        return await self._request("GET", endpoint, "v4", params)
