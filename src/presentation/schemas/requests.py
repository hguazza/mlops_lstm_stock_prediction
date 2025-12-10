"""Request Pydantic schemas for API endpoints."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ConfigOverride(BaseModel):
    """Optional configuration overrides for model training."""

    hidden_size: Optional[int] = Field(
        default=None, ge=16, le=512, description="LSTM hidden size"
    )
    num_layers: Optional[int] = Field(
        default=None, ge=1, le=5, description="Number of LSTM layers"
    )
    dropout: Optional[float] = Field(
        default=None, ge=0.0, le=0.9, description="Dropout rate"
    )
    sequence_length: Optional[int] = Field(
        default=None, ge=10, le=120, description="Input sequence length"
    )
    learning_rate: Optional[float] = Field(
        default=None, gt=0.0, lt=1.0, description="Learning rate"
    )
    batch_size: Optional[int] = Field(
        default=None, ge=8, le=256, description="Batch size"
    )
    epochs: Optional[int] = Field(
        default=None, ge=1, le=1000, description="Number of training epochs"
    )
    early_stopping_patience: Optional[int] = Field(
        default=None, ge=1, le=100, description="Early stopping patience"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "hidden_size": 64,
                "num_layers": 3,
                "epochs": 50,
            }
        }


class TrainRequest(BaseModel):
    """Request schema for training endpoint."""

    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol")
    period: str = Field(
        default="1y",
        description="Historical data period (e.g., '1d', '1mo', '1y', '5y')",
    )
    config: Optional[ConfigOverride] = Field(
        default=None, description="Optional model configuration overrides"
    )

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_uppercase(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    @field_validator("period")
    @classmethod
    def period_must_be_valid(cls, v: str) -> str:
        """Validate period format."""
        valid_periods = [
            "1d",
            "5d",
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
            "ytd",
            "max",
        ]
        if v not in valid_periods:
            raise ValueError(
                f"Invalid period. Must be one of: {', '.join(valid_periods)}"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "period": "1y",
                "config": {
                    "hidden_size": 50,
                    "num_layers": 2,
                    "epochs": 100,
                },
            }
        }


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""

    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol")
    model_version: Optional[str] = Field(
        default="latest",
        description="Model version to use ('latest', 'Production', or version number)",
    )

    @field_validator("symbol")
    @classmethod
    def symbol_must_be_uppercase(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "model_version": "latest",
            }
        }
