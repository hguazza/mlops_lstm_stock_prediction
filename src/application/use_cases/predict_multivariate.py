"""Multivariate stock prediction use case.

Implements complete pipeline for multivariate LSTM predictions:
1. Fetch data for multiple tickers
2. Feature engineering (log-returns + technical indicators)
3. Temporal train/val/test split (walk-forward)
4. Train multivariate LSTM model
5. Validate with directional accuracy
6. Predict with Monte Carlo Dropout
7. Calculate confidence intervals
"""

import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import structlog

from src.application.services.data_service import DataFetchError, DataService
from src.infrastructure.features.technical_indicators import (
    FeatureEngineeringError,
    InsufficientDataError,
    TechnicalIndicatorCalculator,
)
from src.infrastructure.model.lstm_multivariate import MultivariateLSTMPredictor

logger = structlog.get_logger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""

    pass


class PredictionError(Exception):
    """Raised when prediction fails."""

    pass


class PredictMultivariateUseCase:
    """
    Use case for multivariate stock prediction.

    Business logic: Complete multivariate prediction pipeline including:
    1. Multi-ticker data fetching
    2. Feature engineering with technical indicators
    3. Model training with temporal validation
    4. Prediction with uncertainty quantification
    5. Response formatting
    """

    def __init__(
        self,
        data_service: Optional[DataService] = None,
        feature_calculator: Optional[TechnicalIndicatorCalculator] = None,
        mlflow_service=None,
    ):
        """
        Initialize use case.

        Args:
            data_service: DataService instance (creates default if None)
            feature_calculator: TechnicalIndicatorCalculator instance (creates default if None)
            mlflow_service: Optional MLflow service for experiment tracking
        """
        self.data_service = data_service or DataService()
        self.feature_calculator = feature_calculator or TechnicalIndicatorCalculator(
            prevent_leakage=True
        )
        self.mlflow_service = mlflow_service
        self.logger = logger.bind(component="PredictMultivariateUseCase")

    def execute_train_predict(
        self,
        input_tickers: List[str],
        target_ticker: str,
        lookback: int,
        forecast_horizon: int,
        confidence_level: float,
        period: str,
        config_overrides: Optional[Dict] = None,
    ) -> Dict:
        """
        Execute complete training and prediction pipeline.

        Args:
            input_tickers: List of 4 ticker symbols for features
            target_ticker: Target ticker to predict
            lookback: Historical window in days
            forecast_horizon: Forecast horizon in days
            confidence_level: Confidence level for intervals
            period: Historical data period
            config_overrides: Optional model config overrides

        Returns:
            Dictionary with training metrics, prediction, and metadata

        Raises:
            ValidationError: If input validation fails
            DataFetchError: If data fetching fails
            ModelTrainingError: If model training fails
            PredictionError: If prediction fails
        """
        start_time = time.time()

        self.logger.info(
            "executing_multivariate_train_predict",
            input_tickers=input_tickers,
            target_ticker=target_ticker,
            lookback=lookback,
            forecast_horizon=forecast_horizon,
        )

        try:
            # Step 1: Validate inputs
            self._validate_inputs(
                input_tickers, target_ticker, lookback, forecast_horizon
            )

            # Step 2: Fetch data for all tickers
            self.logger.info("step_1_fetching_multi_ticker_data")
            all_tickers = input_tickers + [target_ticker]
            period_days = self._period_to_days(period)

            ticker_data = self.data_service.fetch_multiple_tickers(
                tickers=all_tickers,
                periods=period_days,
                validate=True,
                align_dates=True,
            )

            self.logger.info(
                "data_fetched",
                num_tickers=len(ticker_data),
                common_dates=len(next(iter(ticker_data.values()))),
            )

            # Step 3: Feature engineering
            self.logger.info("step_2_feature_engineering")
            feature_df, feature_names = self.feature_calculator.create_feature_matrix(
                ticker_data=ticker_data,
                target_ticker=target_ticker,
            )

            self.logger.info(
                "features_created",
                num_features=len(feature_names),
                num_samples=len(feature_df),
            )

            # Step 4: Prepare data for training
            if "target" not in feature_df.columns:
                raise FeatureEngineeringError(
                    "Target column not found in feature matrix"
                )

            features = feature_df[feature_names].values
            targets = feature_df["target"].values

            # Step 5: Initialize model
            num_features = len(feature_names)
            model_config = self._build_model_config(
                num_features, lookback, config_overrides
            )

            predictor = MultivariateLSTMPredictor(**model_config)

            # Step 6: Train model with MLflow tracking
            self.logger.info("step_3_training_model")

            if self.mlflow_service:
                self._start_mlflow_run(
                    input_tickers,
                    target_ticker,
                    lookback,
                    forecast_horizon,
                    model_config,
                )

            history = predictor.train(
                features=features,
                targets=targets,
                verbose=True,
                epoch_callback=self._create_mlflow_callback()
                if self.mlflow_service
                else None,
            )

            training_time = time.time() - start_time

            # Step 7: Calculate additional metrics
            train_metrics = self._calculate_metrics(history, training_time)

            # Step 8: Generate prediction with uncertainty
            self.logger.info("step_4_generating_prediction")
            mean_pred, lower_bound, upper_bound = predictor.predict_with_uncertainty(
                features=features,
                n_iterations=100,
                confidence_level=confidence_level,
            )

            # Step 9: Build response
            response = {
                "model_id": f"multivariate_predictor_{target_ticker}",
                "training_metrics": train_metrics,
                "prediction": {
                    "target_ticker": target_ticker,
                    "input_tickers": input_tickers,
                    "predicted_return_pct": float(mean_pred),
                    "confidence_interval": {
                        "lower": float(lower_bound),
                        "upper": float(upper_bound),
                        "confidence_level": confidence_level,
                    },
                    "forecast_horizon_days": forecast_horizon,
                    "prediction_timestamp": datetime.utcnow(),
                    "features_used": feature_names,
                },
                "model_info": {
                    "num_features": num_features,
                    "lookback": lookback,
                    "config": model_config,
                    "trained_at": datetime.utcnow(),
                    "data_rows": len(feature_df),
                },
            }

            # Step 10: Log to MLflow
            if self.mlflow_service:
                self._log_to_mlflow(
                    predictor,
                    train_metrics,
                    input_tickers,
                    target_ticker,
                    features,
                    feature_names,
                )
                response["mlflow_run_id"] = self.mlflow_service.get_last_run_id()

            self.logger.info(
                "multivariate_train_predict_completed",
                target_ticker=target_ticker,
                predicted_return=mean_pred,
                training_time=training_time,
            )

            return response

        except (ValidationError, InsufficientDataError):
            # Re-raise validation errors
            raise
        except DataFetchError as e:
            self.logger.error("data_fetch_failed", error=str(e))
            raise
        except FeatureEngineeringError as e:
            self.logger.error("feature_engineering_failed", error=str(e))
            raise ModelTrainingError(f"Feature engineering failed: {str(e)}") from e
        except Exception as e:
            self.logger.error(
                "unexpected_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ModelTrainingError(f"Training failed: {str(e)}") from e

    def _validate_inputs(
        self,
        input_tickers: List[str],
        target_ticker: str,
        lookback: int,
        forecast_horizon: int,
    ) -> None:
        """Validate use case inputs."""
        # Validate input_tickers
        if not input_tickers or len(input_tickers) != 4:
            raise ValidationError("Must provide exactly 4 input tickers")

        if len(set(input_tickers)) != 4:
            raise ValidationError("All input tickers must be unique")

        # Validate target_ticker
        if not target_ticker or not isinstance(target_ticker, str):
            raise ValidationError("Target ticker must be a non-empty string")

        # Validate lookback
        if not isinstance(lookback, int) or lookback < 20 or lookback > 252:
            raise ValidationError("Lookback must be between 20 and 252 days")

        # Validate forecast_horizon
        if (
            not isinstance(forecast_horizon, int)
            or forecast_horizon < 1
            or forecast_horizon > 30
        ):
            raise ValidationError("Forecast horizon must be between 1 and 30 days")

        self.logger.debug("input_validation_passed")

    def _period_to_days(self, period: str) -> int:
        """Convert period string to number of days."""
        period_map = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "ytd": 365,
            "max": 1825,
        }
        return period_map.get(period, 365)

    def _build_model_config(
        self,
        num_features: int,
        lookback: int,
        config_overrides: Optional[Dict] = None,
    ) -> Dict:
        """Build model configuration."""
        config = {
            "num_features": num_features,
            "lookback": lookback,
            "hidden_size_1": 128,
            "hidden_size_2": 64,
            "dropout": 0.3,
            "use_attention": True,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "early_stopping_patience": 15,
        }

        # Apply overrides
        if config_overrides:
            if "hidden_size" in config_overrides:
                config["hidden_size_1"] = config_overrides["hidden_size"]
            if "num_layers" in config_overrides:
                # Map num_layers to hidden sizes (simplified)
                pass
            if "dropout" in config_overrides:
                config["dropout"] = config_overrides["dropout"]
            if "learning_rate" in config_overrides:
                config["learning_rate"] = config_overrides["learning_rate"]
            if "batch_size" in config_overrides:
                config["batch_size"] = config_overrides["batch_size"]
            if "epochs" in config_overrides:
                config["epochs"] = config_overrides["epochs"]

        return config

    def _calculate_metrics(self, history: Dict, training_time: float) -> Dict:
        """Calculate training metrics from history."""
        mae = history["val_mae"][-1] if "val_mae" in history else 0.0
        rmse = np.sqrt(history["val_loss"][-1])

        return {
            "best_val_loss": float(min(history["val_loss"])),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_val_loss": float(history["val_loss"][-1]),
            "mae": float(mae),
            "rmse": float(rmse),
            "epochs_trained": len(history["train_loss"]),
            "training_time_seconds": training_time,
        }

    def _start_mlflow_run(
        self,
        input_tickers: List[str],
        target_ticker: str,
        lookback: int,
        forecast_horizon: int,
        model_config: Dict,
    ) -> None:
        """Start MLflow run with tags."""
        if not self.mlflow_service:
            return

        try:
            run_name = f"multivariate_{target_ticker}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            tags = {
                "model_type": "MultivariateLSTM",
                "framework": "PyTorch",
                "target_ticker": target_ticker,
                "input_tickers": ",".join(input_tickers),
                "lookback": str(lookback),
                "forecast_horizon": str(forecast_horizon),
            }

            self.mlflow_service.start_run(run_name=run_name, tags=tags)
            self.mlflow_service.log_params(model_config)
            self.mlflow_service.log_params(
                {
                    "input_tickers": ",".join(input_tickers),
                    "target_ticker": target_ticker,
                }
            )
        except Exception as e:
            self.logger.warning("mlflow_start_failed", error=str(e))

    def _create_mlflow_callback(self):
        """Create callback for MLflow logging during training."""

        def callback(**kwargs):
            if not self.mlflow_service:
                return

            event = kwargs.get("event")
            if event == "epoch":
                epoch = kwargs.get("epoch", 0)
                train_loss = kwargs.get("train_loss", 0.0)
                val_loss = kwargs.get("val_loss", 0.0)
                train_mae = kwargs.get("train_mae", 0.0)
                val_mae = kwargs.get("val_mae", 0.0)

                self.mlflow_service.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_mae": train_mae,
                        "val_mae": val_mae,
                    },
                    step=epoch,
                )

        return callback

    def _log_to_mlflow(
        self,
        predictor: MultivariateLSTMPredictor,
        metrics: Dict,
        input_tickers: List[str],
        target_ticker: str,
        features: np.ndarray,
        feature_names: List[str],
    ) -> None:
        """Log model and metrics to MLflow."""
        if not self.mlflow_service:
            return

        try:
            # Log final metrics
            self.mlflow_service.log_metrics(metrics)

            # Log model
            import torch

            sample_input = features[-predictor.lookback :].reshape(
                1, predictor.lookback, predictor.num_features
            )
            sample_input_scaled = predictor.scaler.transform_features(sample_input)

            with torch.no_grad():
                sample_output = (
                    predictor.model(
                        torch.FloatTensor(sample_input_scaled).to(predictor.device)
                    )
                    .cpu()
                    .numpy()
                )

            signature = self.mlflow_service.infer_signature(
                sample_input.reshape(1, -1), sample_output
            )

            registered_model_name = f"multivariate_predictor_{target_ticker}"

            self.mlflow_service.log_model(
                model=predictor.model,
                artifact_path="model",
                signature=signature,
                input_example=sample_input.reshape(1, -1),
                registered_model_name=registered_model_name,
            )

            # Log feature names
            self.mlflow_service.log_params(
                {
                    "feature_names": ",".join(feature_names),
                    "num_features": len(feature_names),
                }
            )

            self.mlflow_service.end_run(status="FINISHED")
            self.logger.info("model_logged_to_mlflow", model_name=registered_model_name)

        except Exception as e:
            self.logger.warning("mlflow_logging_failed", error=str(e))
            try:
                self.mlflow_service.end_run(status="FAILED")
            except Exception:
                pass
