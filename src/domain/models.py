"""Domain models - Pydantic schemas for validation and serialization."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class StockPriceModel(BaseModel):
    """
    Pydantic model for stock price data validation.

    Used for data validation and serialization in API and data processing.
    """

    symbol: str = Field(
        ..., min_length=1, max_length=10, description="Stock ticker symbol"
    )
    date: datetime = Field(..., description="Price timestamp")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    adjusted_close: Optional[float] = Field(
        None, gt=0, description="Adjusted close price"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase and alphanumeric."""
        if not v.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Symbol must be alphanumeric (dots and dashes allowed)")
        return v.upper()

    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float, info) -> float:
        """Ensure high price is greater than or equal to low price."""
        if "low" in info.data and v < info.data["low"]:
            raise ValueError("High price must be >= low price")
        return v

    @field_validator("open")
    @classmethod
    def validate_open(cls, v: float, info) -> float:
        """Ensure open price is within low-high range."""
        if "low" in info.data and "high" in info.data:
            if not (info.data["low"] <= v <= info.data["high"]):
                raise ValueError("Open price must be between low and high")
        return v

    @field_validator("close")
    @classmethod
    def validate_close(cls, v: float, info) -> float:
        """Ensure close price is within low-high range."""
        if "low" in info.data and "high" in info.data:
            if not (info.data["low"] <= v <= info.data["high"]):
                raise ValueError("Close price must be between low and high")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "AAPL",
                "date": "2024-01-15T16:00:00Z",
                "open": 182.15,
                "high": 184.25,
                "low": 181.50,
                "close": 183.89,
                "volume": 50000000,
                "adjusted_close": 183.89,
            }
        }
    }


class PredictionRequest(BaseModel):
    """
    Request model for stock price prediction.

    Attributes:
        symbol: Stock ticker symbol to predict
        periods: Number of historical periods (timesteps) to use for prediction
    """

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g., AAPL, GOOGL)",
        examples=["AAPL", "GOOGL", "MSFT"],
    )
    periods: int = Field(
        default=60,
        ge=10,
        le=252,
        description="Number of historical days to use (10-252)",
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Ensure symbol is uppercase and alphanumeric."""
        if not v.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Symbol must be alphanumeric")
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "AAPL",
                "periods": 60,
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    Response model for stock price prediction.

    Attributes:
        symbol: Stock ticker symbol
        predicted_price: Predicted closing price for next trading day
        confidence_interval: Optional confidence range [lower, upper]
        historical_prices: Recent historical closing prices used for prediction
        prediction_date: When the prediction was made
        model_version: Version of the model used
    """

    symbol: str = Field(..., description="Stock ticker symbol")
    predicted_price: float = Field(
        ..., gt=0, description="Predicted next closing price"
    )
    confidence_interval: Optional[List[float]] = Field(
        None,
        description="95% confidence interval [lower, upper]",
        min_length=2,
        max_length=2,
    )
    historical_prices: List[float] = Field(
        ...,
        description="Recent historical closing prices",
        min_length=1,
    )
    prediction_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of prediction",
    )
    model_version: str = Field(default="1.0.0", description="Model version")

    @field_validator("confidence_interval")
    @classmethod
    def validate_confidence_interval(
        cls, v: Optional[List[float]]
    ) -> Optional[List[float]]:
        """Ensure confidence interval is valid."""
        if v is not None:
            if len(v) != 2:
                raise ValueError("Confidence interval must have exactly 2 values")
            if v[0] >= v[1]:
                raise ValueError("Lower bound must be less than upper bound")
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError("Confidence bounds must be positive")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "AAPL",
                "predicted_price": 185.50,
                "confidence_interval": [182.0, 189.0],
                "historical_prices": [183.89, 184.25, 182.75, 183.15],
                "prediction_date": "2024-01-16T10:30:00Z",
                "model_version": "1.0.0",
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Check timestamp"
    )
    version: str = Field(default="0.1.0", description="API version")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-16T10:30:00Z",
                "version": "0.1.0",
            }
        }
    }
