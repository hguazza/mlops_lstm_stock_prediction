"""Prediction service - Orchestrates model training and prediction."""

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import structlog

from src.infrastructure.model import ModelConfig
from src.infrastructure.model.lstm_model import LSTMStockPredictor

logger = structlog.get_logger(__name__)


class ModelTrainingError(Exception):
    """Raised when model training fails."""

    pass


class PredictionError(Exception):
    """Raised when prediction fails."""

    pass


class PredictionService:
    """
    Service for orchestrating model training and predictions.

    Provides high-level ML operations for use cases,
    abstracting model infrastructure details.
    """

    def __init__(
        self,
        model: Optional[LSTMStockPredictor] = None,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize prediction service.

        Args:
            model: LSTMStockPredictor instance (creates default if None)
            config: Model configuration (uses defaults if None)
        """
        self.config = config or ModelConfig()
        self.model = model or LSTMStockPredictor(config=self.config)
        self.logger = logger.bind(component="PredictionService")
        self._model_metadata: Dict = {}

    def train_model(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        verbose: bool = False,
    ) -> Dict:
        """
        Train LSTM model on stock data.

        Args:
            data: DataFrame with stock prices
            target_column: Column to predict (default: 'Close')
            verbose: Whether to print training progress

        Returns:
            Training history dictionary

        Raises:
            ModelTrainingError: If training fails
        """
        self.logger.info(
            "starting_model_training",
            data_shape=data.shape,
            target_column=target_column,
        )

        try:
            history = self.model.train(
                data=data, target_column=target_column, verbose=verbose
            )

            # Update metadata
            self._model_metadata = {
                "trained_at": datetime.utcnow().isoformat(),
                "data_rows": len(data),
                "target_column": target_column,
                "config": self.config.to_dict(),
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "best_val_loss": min(history["val_loss"]),
                "epochs_trained": len(history["train_loss"]),
            }

            self.logger.info(
                "model_training_completed",
                epochs=len(history["train_loss"]),
                best_val_loss=min(history["val_loss"]),
            )

            return history

        except Exception as e:
            self.logger.error(
                "model_training_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise ModelTrainingError(f"Failed to train model: {str(e)}") from e

    def predict_next_price(
        self, data: pd.DataFrame, target_column: str = "Close"
    ) -> float:
        """
        Predict next price from historical data.

        Args:
            data: DataFrame with historical prices
            target_column: Column to use for prediction

        Returns:
            Predicted next closing price

        Raises:
            PredictionError: If prediction fails
        """
        self.logger.info(
            "predicting_next_price",
            data_rows=len(data),
            target_column=target_column,
        )

        try:
            prediction = self.model.predict_from_dataframe(
                data=data, target_column=target_column
            )

            self.logger.info(
                "prediction_completed",
                predicted_price=prediction,
                last_actual_price=float(data[target_column].iloc[-1]),
            )

            return prediction

        except ValueError as e:
            self.logger.error("prediction_validation_error", error=str(e))
            raise PredictionError(f"Validation error: {str(e)}") from e

        except Exception as e:
            self.logger.error(
                "prediction_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise PredictionError(f"Failed to make prediction: {str(e)}") from e

    def train_and_predict(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        verbose: bool = False,
    ) -> tuple[float, Dict]:
        """
        Train model and immediately make a prediction.

        Convenience method that combines training and prediction.

        Args:
            data: DataFrame with stock prices
            target_column: Column to predict
            verbose: Whether to print training progress

        Returns:
            Tuple of (predicted_price, training_history)

        Raises:
            ModelTrainingError: If training fails
            PredictionError: If prediction fails
        """
        self.logger.info("train_and_predict_pipeline_started")

        # Train model
        history = self.train_model(
            data=data, target_column=target_column, verbose=verbose
        )

        # Make prediction
        prediction = self.predict_next_price(data=data, target_column=target_column)

        self.logger.info(
            "train_and_predict_completed",
            prediction=prediction,
        )

        return prediction, history

    def get_model_info(self) -> Dict:
        """
        Get model metadata and configuration.

        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": "LSTM",
            "config": self.config.to_dict(),
            "is_trained": self.model.is_fitted,
            "metadata": self._model_metadata,
        }

        return info

    def get_historical_predictions(
        self, data: pd.DataFrame, target_column: str = "Close", n_points: int = 10
    ) -> List[float]:
        """
        Get recent historical closing prices.

        Useful for returning context with predictions.

        Args:
            data: DataFrame with historical data
            target_column: Column to extract
            n_points: Number of recent points to return

        Returns:
            List of recent prices
        """
        prices = data[target_column].tail(n_points).tolist()
        return [float(p) for p in prices]

    def calculate_confidence_interval(
        self,
        predicted_price: float,
        historical_data: pd.DataFrame,
        confidence_level: float = 0.95,
    ) -> List[float]:
        """
        Calculate simple confidence interval based on historical volatility.

        This is a simplified approach. More sophisticated methods
        could be implemented using prediction variance, ensemble methods, etc.

        Args:
            predicted_price: Predicted price
            historical_data: Historical price data
            confidence_level: Confidence level (default: 0.95)

        Returns:
            List [lower_bound, upper_bound]
        """
        # Calculate historical volatility
        returns = historical_data["Close"].pct_change().dropna()
        volatility = returns.std()

        # Simple confidence interval (±2 std deviations ≈ 95% confidence)
        margin = 2 * volatility * predicted_price

        lower_bound = max(0, predicted_price - margin)  # Price can't be negative
        upper_bound = predicted_price + margin

        self.logger.info(
            "confidence_interval_calculated",
            predicted_price=predicted_price,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        return [float(lower_bound), float(upper_bound)]

    def save_model(self, save_dir: str) -> None:
        """
        Save model to disk.

        Args:
            save_dir: Directory to save model files

        Raises:
            PredictionError: If save fails
        """
        try:
            self.model.save_model(save_dir)
            self.logger.info("model_saved", path=save_dir)
        except Exception as e:
            self.logger.error("model_save_failed", error=str(e))
            raise PredictionError(f"Failed to save model: {str(e)}") from e

    def load_model(self, load_dir: str) -> None:
        """
        Load model from disk.

        Args:
            load_dir: Directory containing model files

        Raises:
            PredictionError: If load fails
        """
        try:
            self.model.load_model(load_dir)
            self.logger.info("model_loaded", path=load_dir)

            # Update metadata
            self._model_metadata["loaded_from"] = load_dir
            self._model_metadata["loaded_at"] = datetime.utcnow().isoformat()

        except Exception as e:
            self.logger.error("model_load_failed", error=str(e))
            raise PredictionError(f"Failed to load model: {str(e)}") from e
