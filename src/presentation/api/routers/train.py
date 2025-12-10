"""Training router - Model training endpoint."""

import time
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends

from src.application.use_cases.predict_stock import PredictStockPriceUseCase
from src.infrastructure.model import ModelConfig
from src.presentation.api.dependencies import get_predict_stock_price_use_case
from src.presentation.api.errors import TrainingError
from src.presentation.schemas.requests import TrainRequest
from src.presentation.schemas.responses import ModelInfo, TrainResponse, TrainingMetrics

router = APIRouter(prefix="/train", tags=["training"])
logger = structlog.get_logger(__name__)


@router.post("", response_model=TrainResponse, status_code=201)
async def train_model(
    request: TrainRequest,
    use_case: PredictStockPriceUseCase = Depends(get_predict_stock_price_use_case),
) -> TrainResponse:
    """
    Train LSTM model on historical stock data.

    This endpoint:
    - Fetches historical data from yfinance
    - Trains LSTM model with specified/default configuration
    - Logs experiment to MLflow
    - Registers model in MLflow Registry
    - Returns training metrics and model info

    Args:
        request: Training request with symbol, period, and optional config
        use_case: Injected use case dependency

    Returns:
        TrainResponse with model ID, metrics, and MLflow run ID

    Raises:
        HTTPException: If training fails
    """
    logger.info(
        "train_request_received",
        symbol=request.symbol,
        period=request.period,
        has_config_override=request.config is not None,
    )

    start_time = time.time()

    try:
        # Apply config overrides if provided
        config = None
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
                # Recreate model with new config
                from src.infrastructure.model.lstm_model import LSTMStockPredictor

                use_case.prediction_service.model = LSTMStockPredictor(config=config)

        # Execute training through use case
        result = use_case.execute(
            symbol=request.symbol,
            period=request.period,
            train_new_model=True,  # Always train for /train endpoint
        )

        training_time = time.time() - start_time

        # Get MLflow run ID if available
        mlflow_run_id = None
        if use_case.prediction_service.mlflow_service:
            run_info = use_case.prediction_service.mlflow_service.get_run_info()
            if run_info:
                mlflow_run_id = run_info.run_id

        # Build response
        model_id = f"stock_predictor_{request.symbol}"
        if mlflow_run_id:
            model_id = f"{model_id}:latest"

        training_metrics = TrainingMetrics(
            best_val_loss=result.get("best_val_loss", 0.0),
            final_train_loss=result.get("final_train_loss", 0.0),
            final_val_loss=result.get("final_val_loss", 0.0),
            epochs_trained=result.get("epochs_trained", 0),
            training_time_seconds=training_time,
        )

        model_info = ModelInfo(
            model_id=model_id,
            symbol=request.symbol,
            trained_at=datetime.utcnow(),
            data_rows=len(result.get("historical_prices", [])),
            config=use_case.prediction_service.config.to_dict(),
            mlflow_run_id=mlflow_run_id,
            model_stage="Staging",
        )

        logger.info(
            "training_completed",
            symbol=request.symbol,
            epochs=training_metrics.epochs_trained,
            best_val_loss=training_metrics.best_val_loss,
            training_time=training_time,
        )

        return TrainResponse(
            model_id=model_id,
            mlflow_run_id=mlflow_run_id,
            training_metrics=training_metrics,
            model_info=model_info,
        )

    except Exception as e:
        logger.error(
            "training_failed",
            symbol=request.symbol,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise TrainingError(
            message=f"Failed to train model for {request.symbol}: {str(e)}",
            details={"symbol": request.symbol, "error_type": type(e).__name__},
        )
