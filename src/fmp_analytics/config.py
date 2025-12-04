"""Configuration management for FMP Analytics."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # FMP API Configuration
    fmp_api_key: str = Field(..., description="FMP API key for authentication")
    fmp_base_url: str = Field(
        default="https://financialmodelingprep.com/api",
        description="FMP API base URL",
    )

    # OpenAI Configuration (for Agno agent)
    openai_api_key: str = Field(default="", description="OpenAI API key for Agno agent")

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    # Rate limiting
    rate_limit_requests: int = Field(default=300, description="Max requests per minute")
    rate_limit_period: int = Field(default=60, description="Rate limit period in seconds")

    @property
    def fmp_api_url(self) -> str:
        """Get the full FMP API URL with version."""
        return f"{self.fmp_base_url}/v3"

    @property
    def fmp_api_v4_url(self) -> str:
        """Get the FMP API v4 URL."""
        return f"{self.fmp_base_url}/v4"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
