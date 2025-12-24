"""Application services - Orchestrate infrastructure components."""

from typing import Optional

from src.infrastructure.mlflow import MLflowConfig
from src.infrastructure.mlflow.mlflow_service import MLflowService

from .data_service import DataService
from .prediction_service import PredictionService


def create_data_service() -> DataService:
    """
    Factory function to create DataService instance.

    Returns:
        Configured DataService instance
    """
    return DataService()


def create_prediction_service() -> PredictionService:
    """
    Factory function to create PredictionService instance.

    Returns:
        Configured PredictionService instance
    """
    return PredictionService()


def create_prediction_service_with_mlflow(
    mlflow_config: Optional[MLflowConfig] = None,
) -> PredictionService:
    """
    Factory function to create PredictionService with MLflow tracking.

    Args:
        mlflow_config: Optional MLflow configuration (uses defaults if None)

    Returns:
        Configured PredictionService with MLflow tracking enabled
    """
    # Create MLflow configuration from environment variables if not provided
    if mlflow_config is None:
        from src.config import get_config

        app_config = get_config()
        mlflow_config = MLflowConfig(
            tracking_uri=app_config.mlflow.tracking_uri,
            experiment_name=app_config.mlflow.experiment_name,
            artifact_location=app_config.mlflow.artifact_location,
            enable_autolog=app_config.mlflow.enable_autolog,
        )

    # Create and setup MLflow service
    mlflow_service = MLflowService(config=mlflow_config)
    mlflow_service.setup_tracking()

    # Create prediction service with MLflow
    return PredictionService(mlflow_service=mlflow_service)


__all__ = [
    "DataService",
    "PredictionService",
    "create_data_service",
    "create_prediction_service",
    "create_prediction_service_with_mlflow",
]
