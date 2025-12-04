"""Base model classes for FMP Analytics."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class FMPBaseModel(BaseModel):
    """Base model for all FMP data models."""

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="ignore",
    )

    @field_validator("*", mode="before")
    @classmethod
    def parse_none_string(cls, v: Any) -> Any:
        """Convert 'None' strings to None."""
        if v == "None" or v == "":
            return None
        return v


class TimestampedModel(FMPBaseModel):
    """Base model with timestamp fields."""

    @field_validator("*", mode="before")
    @classmethod
    def parse_dates(cls, v: Any, info: Any) -> Any:
        """Parse date strings to date objects."""
        if v is None:
            return v
        field_name = info.field_name if hasattr(info, "field_name") else ""
        if isinstance(v, str) and ("date" in field_name.lower() or field_name in ["timestamp", "time"]):
            try:
                # Try datetime first
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            except ValueError:
                try:
                    # Try date
                    return date.fromisoformat(v[:10])
                except ValueError:
                    return v
        return v
