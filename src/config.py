"""Centralized application configuration."""

import os
from dataclasses import dataclass


@dataclass
class MLflowSettings:
    """MLflow-specific settings."""

    tracking_uri: str = "sqlite:////data/mlflow.db"
    experiment_name: str = "stock_prediction_lstm"
    artifact_location: str = "/data/mlruns_artifacts"
    enable_autolog: bool = True
    registered_model_prefix: str = "stock_predictor"


@dataclass
class ModelSettings:
    """Default model settings."""

    input_size: int = 1
    hidden_size: int = 50
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 60
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class DataSettings:
    """Data loading settings."""

    default_period: str = "60d"
    default_symbol: str = "AAPL"
    yfinance_timeout: int = 30


@dataclass
class AppConfig:
    """
    Application configuration.

    Centralizes all configuration with environment variable overrides.
    """

    mlflow: MLflowSettings
    model: ModelSettings
    data: DataSettings
    environment: str = "development"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Create configuration from environment variables.

        Environment variables:
            MLFLOW_TRACKING_URI: MLflow tracking URI
            MLFLOW_EXPERIMENT_NAME: Experiment name
            MODEL_HIDDEN_SIZE: LSTM hidden size
            MODEL_NUM_LAYERS: LSTM number of layers
            ENVIRONMENT: Application environment (development/staging/production)

        Returns:
            AppConfig instance
        """
        mlflow = MLflowSettings(
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            experiment_name=os.getenv(
                "MLFLOW_EXPERIMENT_NAME", "stock_prediction_lstm"
            ),
            artifact_location=os.getenv(
                "MLFLOW_ARTIFACT_LOCATION", "./mlruns_artifacts"
            ),
            enable_autolog=os.getenv("MLFLOW_ENABLE_AUTOLOG", "true").lower() == "true",
            registered_model_prefix=os.getenv("MLFLOW_MODEL_PREFIX", "stock_predictor"),
        )

        model = ModelSettings(
            hidden_size=int(os.getenv("MODEL_HIDDEN_SIZE", "50")),
            num_layers=int(os.getenv("MODEL_NUM_LAYERS", "2")),
            dropout=float(os.getenv("MODEL_DROPOUT", "0.2")),
            sequence_length=int(os.getenv("MODEL_SEQUENCE_LENGTH", "60")),
            learning_rate=float(os.getenv("MODEL_LEARNING_RATE", "0.001")),
            batch_size=int(os.getenv("MODEL_BATCH_SIZE", "32")),
            epochs=int(os.getenv("MODEL_EPOCHS", "100")),
            early_stopping_patience=int(os.getenv("MODEL_EARLY_STOPPING", "10")),
        )

        data = DataSettings(
            default_period=os.getenv("DATA_DEFAULT_PERIOD", "60d"),
            default_symbol=os.getenv("DATA_DEFAULT_SYMBOL", "AAPL"),
            yfinance_timeout=int(os.getenv("DATA_YFINANCE_TIMEOUT", "30")),
        )

        environment = os.getenv("ENVIRONMENT", "development")

        return cls(
            mlflow=mlflow,
            model=model,
            data=data,
            environment=environment,
        )


def get_config() -> AppConfig:
    """
    Get application configuration.

    Loads from environment variables with sensible defaults.

    Returns:
        AppConfig instance
    """
    return AppConfig.from_env()


# Create default config instance
config = get_config()
