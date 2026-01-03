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
    # #region agent log
    import json

    with open("/data/debug.log", "a") as f:
        f.write(
            json.dumps(
                {
                    "location": "train.py:45",
                    "message": "Train request received",
                    "data": {
                        "symbol": request.symbol,
                        "period": request.period,
                        "period_type": type(request.period).__name__,
                    },
                    "timestamp": time.time() * 1000,
                    "sessionId": "debug-session",
                    "runId": "initial",
                    "hypothesisId": "C_D",
                }
            )
            + "\n"
        )
    # #endregion

    start_time = time.time()

    # Convert period string to number of days
    # #region agent log
    period_days_map = {
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
    periods_days = period_days_map.get(request.period, 365)
    import json

    with open("/data/debug.log", "a") as f:
        f.write(
            json.dumps(
                {
                    "location": "train.py:58",
                    "message": "Period conversion",
                    "data": {
                        "period_str": request.period,
                        "periods_days": periods_days,
                    },
                    "timestamp": time.time() * 1000,
                    "sessionId": "debug-session",
                    "runId": "post-fix",
                    "hypothesisId": "FIX",
                }
            )
            + "\n"
        )
    # #endregion

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
        # #region agent log
        import json

        with open("/data/debug.log", "a") as f:
            f.write(
                json.dumps(
                    {
                        "location": "train.py:103",
                        "message": "Before use_case.execute call",
                        "data": {
                            "symbol": request.symbol,
                            "periods_days": periods_days,
                            "use_case_type": type(use_case).__name__,
                        },
                        "timestamp": time.time() * 1000,
                        "sessionId": "debug-session",
                        "runId": "post-fix",
                        "hypothesisId": "FIX",
                    }
                )
                + "\n"
            )
        # #endregion
        result = use_case.execute(
            symbol=request.symbol,
            periods=periods_days,  # Fixed: use 'periods' (plural) and pass days as int
            include_confidence=True,
        )

        training_time = time.time() - start_time

        # #region agent log
        import json

        with open("/data/debug.log", "a") as f:
            f.write(
                json.dumps(
                    {
                        "location": "train.py:112",
                        "message": "After use_case.execute",
                        "data": {
                            "result_type": type(result).__name__,
                            "has_predicted_price": hasattr(result, "predicted_price"),
                            "has_get_method": hasattr(result, "get"),
                        },
                        "timestamp": time.time() * 1000,
                        "sessionId": "debug-session",
                        "runId": "post-fix",
                        "hypothesisId": "F_G",
                    }
                )
                + "\n"
            )
        # #endregion

        # Get training metrics from model_info
        model_info_dict = use_case.prediction_service.get_model_info()

        # #region agent log
        with open("/data/debug.log", "a") as f:
            f.write(
                json.dumps(
                    {
                        "location": "train.py:120",
                        "message": "Model info retrieved",
                        "data": {
                            "model_info_keys": list(model_info_dict.keys())
                            if isinstance(model_info_dict, dict)
                            else "not_dict",
                            "metadata_keys": list(
                                model_info_dict.get("metadata", {}).keys()
                            )
                            if isinstance(model_info_dict, dict)
                            else "N/A",
                        },
                        "timestamp": time.time() * 1000,
                        "sessionId": "debug-session",
                        "runId": "post-fix",
                        "hypothesisId": "G",
                    }
                )
                + "\n"
            )
        # #endregion

        # Get MLflow run ID from the response (captured during training)
        mlflow_run_id = getattr(result, "_mlflow_run_id", None)

        # Fallback: try to get from service if not in response
        if mlflow_run_id is None and use_case.prediction_service.mlflow_service:
            mlflow_run_id = use_case.prediction_service.mlflow_service.get_last_run_id()

        logger.info("final_mlflow_run_id", run_id=mlflow_run_id)

        # Build response
        model_id = f"stock_predictor_{request.symbol}"
        if mlflow_run_id:
            model_id = f"{model_id}:latest"

        # Extract training metrics from model_info metadata
        metadata = model_info_dict.get("metadata", {})
        training_metrics = TrainingMetrics(
            best_val_loss=metadata.get("best_val_loss", 0.0),
            final_train_loss=metadata.get("final_train_loss", 0.0),
            final_val_loss=metadata.get("final_val_loss", 0.0),
            epochs_trained=metadata.get("epochs_trained", 0),
            training_time_seconds=training_time,
        )

        model_info = ModelInfo(
            model_id=model_id,
            symbol=request.symbol,
            trained_at=datetime.utcnow(),
            data_rows=len(result.historical_prices)
            if hasattr(result, "historical_prices")
            else 0,
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
        # #region agent log
        import json

        with open("/data/debug.log", "a") as f:
            f.write(
                json.dumps(
                    {
                        "location": "train.py:140",
                        "message": "Exception caught in train router",
                        "data": {
                            "error_str": str(e),
                            "error_type": type(e).__name__,
                            "symbol": request.symbol,
                            "period_value": request.period,
                        },
                        "timestamp": time.time() * 1000,
                        "sessionId": "debug-session",
                        "runId": "initial",
                        "hypothesisId": "A_C_E",
                    }
                )
                + "\n"
            )
        # #endregion
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
