"""Response Pydantic schemas for API endpoints."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .common import StatusEnum


class TrainingMetrics(BaseModel):
    """Training metrics."""

    best_val_loss: float = Field(..., description="Best validation loss achieved")
    final_train_loss: float = Field(..., description="Final training loss")
    final_val_loss: float = Field(..., description="Final validation loss")
    epochs_trained: int = Field(..., description="Number of epochs trained")
    training_time_seconds: Optional[float] = Field(
        default=None, description="Training time in seconds"
    )


class ModelInfo(BaseModel):
    """Model information."""

    model_id: str = Field(..., description="Model identifier (name:version)")
    symbol: str = Field(..., description="Stock symbol")
    trained_at: datetime = Field(..., description="Training timestamp")
    data_rows: int = Field(..., description="Number of data rows used for training")
    config: Dict[str, Any] = Field(..., description="Model configuration")
    mlflow_run_id: Optional[str] = Field(default=None, description="MLflow run ID")
    model_stage: Optional[str] = Field(default=None, description="Model registry stage")


class TrainResponse(BaseModel):
    """Response for training endpoint."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS)
    model_id: str = Field(..., description="Trained model identifier")
    mlflow_run_id: Optional[str] = Field(default=None, description="MLflow run ID")
    training_metrics: TrainingMetrics = Field(..., description="Training metrics")
    model_info: ModelInfo = Field(..., description="Model information")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "model_id": "stock_predictor_AAPL:3",
                "mlflow_run_id": "abc123def456",
                "training_metrics": {
                    "best_val_loss": 0.0234,
                    "final_train_loss": 0.0189,
                    "final_val_loss": 0.0245,
                    "epochs_trained": 45,
                    "training_time_seconds": 120.5,
                },
                "model_info": {
                    "model_id": "stock_predictor_AAPL:3",
                    "symbol": "AAPL",
                    "trained_at": "2024-12-10T14:30:52Z",
                    "data_rows": 252,
                    "config": {
                        "hidden_size": 50,
                        "num_layers": 2,
                        "epochs": 100,
                    },
                    "mlflow_run_id": "abc123def456",
                },
            }
        }


class ConfidenceInterval(BaseModel):
    """Confidence interval for prediction."""

    lower: float = Field(..., description="Lower bound")
    upper: float = Field(..., description="Upper bound")
    confidence_level: float = Field(default=0.95, description="Confidence level (0-1)")


class PredictionDetails(BaseModel):
    """Prediction details."""

    symbol: str = Field(..., description="Stock symbol")
    predicted_price: float = Field(..., description="Predicted next closing price")
    confidence_interval: ConfidenceInterval = Field(
        ..., description="95% confidence interval"
    )
    prediction_date: str = Field(..., description="Date of prediction")
    current_price: Optional[float] = Field(
        default=None, description="Current/latest price"
    )


class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS)
    prediction: PredictionDetails = Field(..., description="Prediction details")
    model_info: ModelInfo = Field(..., description="Model information")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "prediction": {
                    "symbol": "AAPL",
                    "predicted_price": 178.45,
                    "confidence_interval": {
                        "lower": 175.20,
                        "upper": 181.70,
                        "confidence_level": 0.95,
                    },
                    "prediction_date": "2024-12-11",
                    "current_price": 176.80,
                },
                "model_info": {
                    "model_id": "stock_predictor_AAPL:3",
                    "symbol": "AAPL",
                    "trained_at": "2024-12-10T14:30:52Z",
                    "data_rows": 252,
                    "config": {"hidden_size": 50, "num_layers": 2},
                    "model_stage": "Production",
                },
            }
        }


class TrainPredictResponse(BaseModel):
    """Response for train-predict endpoint (combined pipeline)."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS)
    model_id: str = Field(..., description="Trained model identifier")
    mlflow_run_id: Optional[str] = Field(default=None, description="MLflow run ID")
    training_metrics: TrainingMetrics = Field(..., description="Training metrics")
    prediction: PredictionDetails = Field(..., description="Prediction details")
    model_info: ModelInfo = Field(..., description="Model information")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "model_id": "stock_predictor_AAPL:4",
                "mlflow_run_id": "xyz789abc123",
                "training_metrics": {
                    "best_val_loss": 0.0198,
                    "final_train_loss": 0.0175,
                    "final_val_loss": 0.0201,
                    "epochs_trained": 52,
                    "training_time_seconds": 135.2,
                },
                "prediction": {
                    "symbol": "AAPL",
                    "predicted_price": 179.20,
                    "confidence_interval": {
                        "lower": 176.10,
                        "upper": 182.30,
                        "confidence_level": 0.95,
                    },
                    "prediction_date": "2024-12-11",
                    "current_price": 177.50,
                },
                "model_info": {
                    "model_id": "stock_predictor_AAPL:4",
                    "symbol": "AAPL",
                    "trained_at": "2024-12-10T15:22:10Z",
                    "data_rows": 252,
                    "config": {"hidden_size": 50, "num_layers": 2},
                },
            }
        }


class ModelSummary(BaseModel):
    """Summary of a registered model."""

    symbol: str = Field(..., description="Stock symbol")
    versions: int = Field(..., description="Total number of versions")
    latest_version: str = Field(..., description="Latest version number")
    production_version: Optional[str] = Field(
        default=None, description="Current production version"
    )
    last_trained: Optional[datetime] = Field(
        default=None, description="Last training timestamp"
    )


class ModelsListResponse(BaseModel):
    """Response for listing models endpoint."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS)
    models: List[ModelSummary] = Field(..., description="List of registered models")
    total_models: int = Field(..., description="Total number of models")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "total_models": 2,
                "models": [
                    {
                        "symbol": "AAPL",
                        "versions": 5,
                        "latest_version": "5",
                        "production_version": "3",
                        "last_trained": "2024-12-10T14:30:52Z",
                    },
                    {
                        "symbol": "GOOGL",
                        "versions": 2,
                        "latest_version": "2",
                        "production_version": None,
                        "last_trained": "2024-12-09T10:15:20Z",
                    },
                ],
            }
        }


class ModelMetrics(BaseModel):
    """Model performance metrics."""

    best_val_loss: Optional[float] = Field(default=None)
    final_train_loss: Optional[float] = Field(default=None)
    final_val_loss: Optional[float] = Field(default=None)
    epochs_trained: Optional[int] = Field(default=None)


class ModelInfoResponse(BaseModel):
    """Response for model info endpoint."""

    status: StatusEnum = Field(default=StatusEnum.SUCCESS)
    model_info: ModelInfo = Field(..., description="Detailed model information")
    metrics: Optional[ModelMetrics] = Field(
        default=None, description="Model performance metrics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "model_info": {
                    "model_id": "stock_predictor_AAPL:5",
                    "symbol": "AAPL",
                    "trained_at": "2024-12-10T14:30:52Z",
                    "data_rows": 252,
                    "config": {
                        "hidden_size": 50,
                        "num_layers": 2,
                        "epochs": 100,
                    },
                    "mlflow_run_id": "abc123def456",
                    "model_stage": "Staging",
                },
                "metrics": {
                    "best_val_loss": 0.0234,
                    "final_train_loss": 0.0189,
                    "final_val_loss": 0.0245,
                    "epochs_trained": 45,
                },
            }
        }
