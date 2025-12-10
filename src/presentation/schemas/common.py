"""Common Pydantic schemas used across API."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class StatusEnum(str, Enum):
    """Status enumeration for API responses."""

    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ErrorResponse(BaseModel):
    """Standard error response format."""

    status: str = Field(default="error", description="Response status")
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error": "ValidationError",
                "message": "Invalid stock symbol format",
                "details": {
                    "symbol": "INVALID!",
                    "expected_format": "uppercase letters",
                },
                "timestamp": "2024-12-10T14:30:52.123456",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: StatusEnum = Field(..., description="Overall health status")
    version: str = Field(default="0.1.0", description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )
    dependencies: Dict[str, str] = Field(
        ..., description="Status of external dependencies"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2024-12-10T14:30:52.123456",
                "dependencies": {
                    "mlflow": "connected",
                    "yfinance": "accessible",
                },
            }
        }
