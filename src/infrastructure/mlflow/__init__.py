"""MLflow infrastructure - Experiment tracking and model registry."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MLflowConfig:
    """
    Configuration for MLflow experiment tracking.

    Attributes:
        tracking_uri: URI for MLflow tracking backend (SQLite by default)
        experiment_name: Name of the experiment
        artifact_location: Path for storing artifacts
        enable_autolog: Enable PyTorch autologging
        log_model_signature: Enable model signature inference
        registered_model_name: Name for model registry (None = auto-generate)
    """

    tracking_uri: str = "sqlite:////data/mlflow.db"
    experiment_name: str = "stock_prediction_lstm"
    artifact_location: str = "/data/mlruns_artifacts"
    enable_autolog: bool = True
    log_model_signature: bool = True
    registered_model_name: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.tracking_uri:
            raise ValueError("tracking_uri cannot be empty")
        if not self.experiment_name:
            raise ValueError("experiment_name cannot be empty")


def get_default_config() -> MLflowConfig:
    """
    Get default MLflow configuration.

    Returns:
        MLflowConfig with default values
    """
    return MLflowConfig()
