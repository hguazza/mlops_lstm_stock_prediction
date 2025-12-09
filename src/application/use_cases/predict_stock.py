"""Predict stock price use case."""

from datetime import datetime
from typing import Optional

import structlog

from src.application.services.data_service import DataFetchError, DataService
from src.application.services.prediction_service import (
    ModelTrainingError,
    PredictionError,
    PredictionService,
)
from src.domain.models import PredictionResponse
from src.infrastructure.model import ModelConfig

logger = structlog.get_logger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class PredictStockPriceUseCase:
    """
    Use case for predicting stock prices.

    Business logic: Complete prediction pipeline including:
    1. Data fetching
    2. Model training
    3. Prediction generation
    4. Response formatting
    """

    def __init__(
        self,
        data_service: Optional[DataService] = None,
        prediction_service: Optional[PredictionService] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        """
        Initialize use case.

        Args:
            data_service: DataService instance (creates default if None)
            prediction_service: PredictionService instance (creates default if None)
            model_config: Model configuration (uses defaults if None)
        """
        self.data_service = data_service or DataService()
        self.model_config = model_config or ModelConfig()
        self.prediction_service = prediction_service or PredictionService(
            config=self.model_config
        )
        self.logger = logger.bind(component="PredictStockPriceUseCase")

    def execute(
        self,
        symbol: str,
        periods: int = 60,
        include_confidence: bool = True,
    ) -> PredictionResponse:
        """
        Execute the use case to predict stock price.

        Full pipeline:
        1. Validate inputs
        2. Fetch historical data
        3. Train model on data
        4. Generate prediction
        5. Calculate confidence interval
        6. Return formatted response

        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            periods: Number of historical days to use (default: 60)
            include_confidence: Whether to calculate confidence interval

        Returns:
            PredictionResponse with prediction and metadata

        Raises:
            ValidationError: If input validation fails
            DataFetchError: If data fetching fails
            ModelTrainingError: If model training fails
            PredictionError: If prediction fails
        """
        self.logger.info(
            "executing_predict_stock_price",
            symbol=symbol,
            periods=periods,
        )

        # Step 1: Validate inputs
        self._validate_inputs(symbol, periods)

        try:
            # Step 2: Fetch historical data
            self.logger.info("step_1_fetching_data", symbol=symbol, periods=periods)

            # Fetch more data for better training (2x the prediction window)
            training_periods = max(120, periods * 2)
            data = self.data_service.prepare_data_for_training(
                symbol=symbol, periods=training_periods
            )

            self.logger.info(
                "data_fetched",
                symbol=symbol,
                rows=len(data),
                date_range=f"{data.index[0]} to {data.index[-1]}",
            )

            # Step 3: Train model
            self.logger.info("step_2_training_model", symbol=symbol)
            history = self.prediction_service.train_model(
                data=data, target_column="Close", verbose=False
            )

            self.logger.info(
                "model_trained",
                epochs=len(history["train_loss"]),
                final_val_loss=history["val_loss"][-1],
            )

            # Step 4: Generate prediction
            self.logger.info("step_3_generating_prediction", symbol=symbol)
            predicted_price = self.prediction_service.predict_next_price(
                data=data, target_column="Close"
            )

            self.logger.info("prediction_generated", predicted_price=predicted_price)

            # Step 5: Get historical prices for context
            historical_prices = self.prediction_service.get_historical_predictions(
                data=data, target_column="Close", n_points=min(10, len(data))
            )

            # Step 6: Calculate confidence interval if requested
            confidence_interval = None
            if include_confidence:
                confidence_interval = (
                    self.prediction_service.calculate_confidence_interval(
                        predicted_price=predicted_price,
                        historical_data=data,
                        confidence_level=0.95,
                    )
                )
                self.logger.info(
                    "confidence_interval_calculated",
                    interval=confidence_interval,
                )

            # Step 7: Get model metadata
            model_info = self.prediction_service.get_model_info()

            # Step 8: Create and return response
            response = PredictionResponse(
                symbol=symbol.upper(),
                predicted_price=predicted_price,
                confidence_interval=confidence_interval,
                historical_prices=historical_prices,
                prediction_date=datetime.utcnow(),
                model_version=f"{model_info['config']['hidden_size']}-{model_info['config']['num_layers']}",
            )

            self.logger.info(
                "predict_stock_price_completed",
                symbol=symbol,
                predicted_price=predicted_price,
            )

            return response

        except DataFetchError:
            # Re-raise data errors
            self.logger.error("data_fetch_failed", symbol=symbol)
            raise

        except ModelTrainingError:
            # Re-raise training errors
            self.logger.error("model_training_failed", symbol=symbol)
            raise

        except PredictionError:
            # Re-raise prediction errors
            self.logger.error("prediction_failed", symbol=symbol)
            raise

        except Exception as e:
            self.logger.error(
                "unexpected_error_in_use_case",
                symbol=symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise PredictionError(f"Unexpected error: {str(e)}") from e

    def execute_with_existing_model(
        self,
        symbol: str,
        periods: int = 60,
        model_path: Optional[str] = None,
    ) -> PredictionResponse:
        """
        Execute prediction using pre-trained model.

        Useful when model is already trained and saved.

        Args:
            symbol: Stock ticker symbol
            periods: Number of historical days to fetch
            model_path: Path to saved model (optional)

        Returns:
            PredictionResponse

        Raises:
            ValidationError: If validation fails
            DataFetchError: If data fetching fails
            PredictionError: If prediction fails
        """
        self.logger.info(
            "executing_with_existing_model",
            symbol=symbol,
            model_path=model_path,
        )

        # Validate inputs
        self._validate_inputs(symbol, periods)

        # Load model if path provided
        if model_path:
            self.prediction_service.load_model(model_path)

        # Fetch data
        data = self.data_service.get_latest_prices(symbol=symbol, periods=periods)

        # Predict
        predicted_price = self.prediction_service.predict_next_price(data=data)

        # Get context
        historical_prices = self.prediction_service.get_historical_predictions(
            data=data, n_points=min(10, len(data))
        )

        # Create response
        response = PredictionResponse(
            symbol=symbol.upper(),
            predicted_price=predicted_price,
            historical_prices=historical_prices,
            prediction_date=datetime.utcnow(),
            model_version="loaded",
        )

        return response

    def _validate_inputs(self, symbol: str, periods: int) -> None:
        """
        Validate use case inputs.

        Args:
            symbol: Stock ticker symbol
            periods: Number of periods

        Raises:
            ValidationError: If validation fails
        """
        # Validate symbol
        if not symbol or not isinstance(symbol, str):
            raise ValidationError("Symbol must be a non-empty string")

        symbol = symbol.strip()
        if not symbol:
            raise ValidationError("Symbol cannot be empty or whitespace")

        if len(symbol) > 10:
            raise ValidationError("Symbol too long (max 10 characters)")

        # Validate periods
        if not isinstance(periods, int):
            raise ValidationError("Periods must be an integer")

        if periods < 10:
            raise ValidationError(
                f"Periods must be at least 10 (got {periods}). "
                "Need sufficient historical data for prediction."
            )

        if periods > 500:
            raise ValidationError(
                f"Periods cannot exceed 500 (got {periods}). "
                "Use a smaller window for better performance."
            )

        self.logger.debug("input_validation_passed", symbol=symbol, periods=periods)
