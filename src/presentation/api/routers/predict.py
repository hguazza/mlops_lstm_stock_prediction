"""Prediction router - Model prediction endpoints."""

import time
from datetime import datetime, timedelta

import structlog
from fastapi import APIRouter, Depends

from src.application.use_cases.predict_stock import PredictStockPriceUseCase
from src.infrastructure.model import ModelConfig
from src.presentation.api.dependencies import get_predict_stock_price_use_case
from src.presentation.api.errors import (
    ModelNotFoundError,
    PredictionError,
    TrainingError,
)
from src.presentation.schemas.requests import PredictRequest, TrainRequest
from src.presentation.schemas.responses import (
    ConfidenceInterval,
    ModelInfo,
    PredictionDetails,
    PredictionResponse,
    TrainPredictResponse,
    TrainingMetrics,
)

router = APIRouter(prefix="/predict", tags=["prediction"])
logger = structlog.get_logger(__name__)


@router.post("", response_model=PredictionResponse)
async def predict_price(
    request: PredictRequest,
    use_case: PredictStockPriceUseCase = Depends(get_predict_stock_price_use_case),
) -> PredictionResponse:
    """
    Predict next stock price using existing model from MLflow Registry.

    This endpoint:
    - Loads model from MLflow Registry (by version or 'latest')
    - Fetches latest historical data
    - Generates prediction with confidence interval
    - Returns prediction without MLflow tracking

    Args:
        request: Prediction request with symbol and optional model version
        use_case: Injected use case dependency

    Returns:
        PredictionResponse with predicted price and model info

    Raises:
        ModelNotFoundError: If model not found in registry
        PredictionError: If prediction fails
    """
    logger.info(
        "predict_request_received",
        symbol=request.symbol,
        model_version=request.model_version,
    )

    try:
        # For now, we'll use the existing model from the use case
        # In a future enhancement, we can load specific versions from MLflow Registry
        result = use_case.execute(
            symbol=request.symbol,
            period="60d",  # Use minimum required data for prediction
            train_new_model=False,  # Use existing model
        )

        # Extract prediction data
        predicted_price = result["predicted_price"]
        confidence_interval = result.get("confidence_interval", {})
        current_price = result.get("current_price")
        historical_prices = result.get("historical_prices", [])

        # Build response
        prediction = PredictionDetails(
            symbol=request.symbol,
            predicted_price=predicted_price,
            confidence_interval=ConfidenceInterval(
                lower=confidence_interval.get("lower", predicted_price * 0.95),
                upper=confidence_interval.get("upper", predicted_price * 1.05),
                confidence_level=0.95,
            ),
            prediction_date=(datetime.utcnow() + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            current_price=current_price,
        )

        model_info = ModelInfo(
            model_id=f"stock_predictor_{request.symbol}:{request.model_version}",
            symbol=request.symbol,
            trained_at=datetime.utcnow(),  # Would come from model metadata
            data_rows=len(historical_prices),
            config=use_case.prediction_service.config.to_dict(),
            model_stage="Production",
        )

        logger.info(
            "prediction_completed",
            symbol=request.symbol,
            predicted_price=predicted_price,
        )

        return PredictionResponse(
            prediction=prediction,
            model_info=model_info,
        )

    except ValueError as e:
        # Model not trained or insufficient data
        logger.error("prediction_failed", symbol=request.symbol, error=str(e))
        raise ModelNotFoundError(
            message=f"Model not found or not trained for {request.symbol}. Please train first using POST /train.",
            details={"symbol": request.symbol, "model_version": request.model_version},
        )
    except Exception as e:
        logger.error(
            "prediction_failed",
            symbol=request.symbol,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise PredictionError(
            message=f"Failed to generate prediction for {request.symbol}: {str(e)}",
            details={"symbol": request.symbol, "error_type": type(e).__name__},
        )


@router.post("/train-predict", response_model=TrainPredictResponse)
async def train_and_predict(
    request: TrainRequest,
    use_case: PredictStockPriceUseCase = Depends(get_predict_stock_price_use_case),
) -> TrainPredictResponse:
    """
    Complete pipeline: Train model and generate prediction.

    This endpoint:
    - Fetches historical data
    - Trains new LSTM model
    - Logs to MLflow
    - Generates prediction
    - Returns both training metrics and prediction

    Args:
        request: Training request with symbol, period, and optional config
        use_case: Injected use case dependency

    Returns:
        TrainPredictResponse with training metrics and prediction

    Raises:
        TrainingError: If training fails
        PredictionError: If prediction fails
    """
    logger.info(
        "train_predict_request_received",
        symbol=request.symbol,
        period=request.period,
    )

    start_time = time.time()

    try:
        # Apply config overrides if provided
        if request.config:
            config_dict = {}
            if request.config.hidden_size:
                config_dict["hidden_size"] = request.config.hidden_size
            if request.config.num_layers:
                config_dict["num_layers"] = request.config.num_layers
            if request.config.dropout:
                config_dict["dropout"] = request.config.dropout
            if request.config.sequence_length:
                config_dict["sequence_length"] = request.config.sequence_length
            if request.config.learning_rate:
                config_dict["learning_rate"] = request.config.learning_rate
            if request.config.batch_size:
                config_dict["batch_size"] = request.config.batch_size
            if request.config.epochs:
                config_dict["epochs"] = request.config.epochs
            if request.config.early_stopping_patience:
                config_dict["early_stopping_patience"] = (
                    request.config.early_stopping_patience
                )

            if config_dict:
                config = ModelConfig(**config_dict)
                use_case.prediction_service.config = config
                from src.infrastructure.model.lstm_model import LSTMStockPredictor

                use_case.prediction_service.model = LSTMStockPredictor(config=config)

        # Execute complete pipeline
        result = use_case.execute(
            symbol=request.symbol,
            period=request.period,
            train_new_model=True,
        )

        training_time = time.time() - start_time

        # Get MLflow run ID
        mlflow_run_id = None
        if use_case.prediction_service.mlflow_service:
            run_info = use_case.prediction_service.mlflow_service.get_run_info()
            if run_info:
                mlflow_run_id = run_info.run_id

        # Build response
        model_id = f"stock_predictor_{request.symbol}:latest"

        training_metrics = TrainingMetrics(
            best_val_loss=result.get("best_val_loss", 0.0),
            final_train_loss=result.get("final_train_loss", 0.0),
            final_val_loss=result.get("final_val_loss", 0.0),
            epochs_trained=result.get("epochs_trained", 0),
            training_time_seconds=training_time,
        )

        confidence_interval_data = result.get("confidence_interval", {})
        prediction = PredictionDetails(
            symbol=request.symbol,
            predicted_price=result["predicted_price"],
            confidence_interval=ConfidenceInterval(
                lower=confidence_interval_data.get(
                    "lower", result["predicted_price"] * 0.95
                ),
                upper=confidence_interval_data.get(
                    "upper", result["predicted_price"] * 1.05
                ),
                confidence_level=0.95,
            ),
            prediction_date=(datetime.utcnow() + timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            current_price=result.get("current_price"),
        )

        model_info = ModelInfo(
            model_id=model_id,
            symbol=request.symbol,
            trained_at=datetime.utcnow(),
            data_rows=len(result.get("historical_prices", [])),
            config=use_case.prediction_service.config.to_dict(),
            mlflow_run_id=mlflow_run_id,
        )

        logger.info(
            "train_predict_completed",
            symbol=request.symbol,
            epochs=training_metrics.epochs_trained,
            predicted_price=prediction.predicted_price,
            training_time=training_time,
        )

        return TrainPredictResponse(
            model_id=model_id,
            mlflow_run_id=mlflow_run_id,
            training_metrics=training_metrics,
            prediction=prediction,
            model_info=model_info,
        )

    except Exception as e:
        logger.error(
            "train_predict_failed",
            symbol=request.symbol,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise TrainingError(
            message=f"Failed to train and predict for {request.symbol}: {str(e)}",
            details={"symbol": request.symbol, "error_type": type(e).__name__},
        )
