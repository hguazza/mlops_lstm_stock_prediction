"""Multivariate prediction router - Endpoints for multivariate LSTM predictions."""

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException

from src.application.use_cases.predict_multivariate import (
    ModelTrainingError,
    PredictMultivariateUseCase,
    ValidationError,
)
from src.presentation.schemas.requests import (
    MultivariatePredictRequest,
    MultivariateTrainPredictRequest,
)
from src.presentation.schemas.responses import (
    ConfidenceInterval,
    ModelInfo,
    MultivariatePredictionDetails,
    MultivariatePredictionResponse,
    MultivariateTrainMetrics,
    MultivariateTrainPredictResponse,
)

router = APIRouter(prefix="/multivariate", tags=["multivariate-prediction"])
logger = structlog.get_logger(__name__)


def get_multivariate_use_case() -> PredictMultivariateUseCase:
    """
    Dependency injection for multivariate use case.

    Creates a new instance with MLflow tracking enabled.
    """
    from src.infrastructure.mlflow.mlflow_service import MLflowService

    try:
        mlflow_service = MLflowService()
        return PredictMultivariateUseCase(mlflow_service=mlflow_service)
    except Exception as e:
        logger.warning("mlflow_service_unavailable", error=str(e))
        return PredictMultivariateUseCase(mlflow_service=None)


@router.post("/train-predict", response_model=MultivariateTrainPredictResponse)
async def train_and_predict_multivariate(
    request: MultivariateTrainPredictRequest,
    use_case: PredictMultivariateUseCase = Depends(get_multivariate_use_case),
) -> MultivariateTrainPredictResponse:
    """
    Train multivariate LSTM model and generate prediction.

    This endpoint implements the complete pipeline:
    1. Fetches historical data for 4 input tickers + target ticker
    2. Calculates features: log-returns + technical indicators (RSI, MACD, volatility, volume)
    3. Trains multivariate LSTM with attention mechanism
    4. Generates prediction with Monte Carlo Dropout for uncertainty quantification
    5. Returns training metrics, prediction, and confidence intervals

    **Input Features (24 total for 4 tickers):**
    - Log-returns (stationarity)
    - RSI (momentum)
    - MACD histogram (trend)
    - Realized volatility (risk)
    - Normalized volume (activity)

    **Output:**
    - Predicted return in percentage (e.g., 2.5 = +2.5%)
    - Confidence interval (95% default)
    - Training metrics (MAE, RMSE, directional accuracy)

    **Example Request:**
    ```json
    {
        "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
        "target_ticker": "NVDA",
        "lookback": 60,
        "forecast_horizon": 5,
        "confidence_level": 0.95,
        "period": "1y"
    }
    ```

    **Example Response:**
    ```json
    {
        "status": "success",
        "model_id": "multivariate_predictor_NVDA:1",
        "training_metrics": {
            "mae": 0.0234,
            "rmse": 0.0312,
            "directional_accuracy": 0.64,
            "epochs_trained": 45
        },
        "prediction": {
            "target_ticker": "NVDA",
            "predicted_return_pct": 2.34,
            "confidence_interval": {"lower": 0.5, "upper": 4.2}
        }
    }
    ```

    Args:
        request: Multivariate training and prediction request
        use_case: Injected use case dependency

    Returns:
        MultivariateTrainPredictResponse with training metrics and prediction

    Raises:
        HTTPException 400: If validation fails (invalid tickers, parameters)
        HTTPException 500: If training or prediction fails
    """
    logger.info(
        "multivariate_train_predict_request_received",
        input_tickers=request.input_tickers,
        target_ticker=request.target_ticker,
        lookback=request.lookback,
        forecast_horizon=request.forecast_horizon,
    )

    start_time = time.time()

    try:
        # Extract config overrides if provided
        config_overrides = None
        if request.config:
            config_overrides = {
                "hidden_size": request.config.hidden_size,
                "num_layers": request.config.num_layers,
                "dropout": request.config.dropout,
                "learning_rate": request.config.learning_rate,
                "batch_size": request.config.batch_size,
                "epochs": request.config.epochs,
            }
            # Remove None values
            config_overrides = {
                k: v for k, v in config_overrides.items() if v is not None
            }

        # Execute use case
        result = use_case.execute_train_predict(
            input_tickers=request.input_tickers,
            target_ticker=request.target_ticker,
            lookback=request.lookback,
            forecast_horizon=request.forecast_horizon,
            confidence_level=request.confidence_level,
            period=request.period,
            config_overrides=config_overrides,
        )

        # Build response
        response = MultivariateTrainPredictResponse(
            model_id=result["model_id"],
            mlflow_run_id=result.get("mlflow_run_id"),
            training_metrics=MultivariateTrainMetrics(
                best_val_loss=result["training_metrics"]["best_val_loss"],
                final_train_loss=result["training_metrics"]["final_train_loss"],
                final_val_loss=result["training_metrics"]["final_val_loss"],
                mae=result["training_metrics"]["mae"],
                rmse=result["training_metrics"]["rmse"],
                directional_accuracy=result["training_metrics"].get(
                    "directional_accuracy"
                ),
                epochs_trained=result["training_metrics"]["epochs_trained"],
                training_time_seconds=result["training_metrics"][
                    "training_time_seconds"
                ],
            ),
            prediction=MultivariatePredictionDetails(
                target_ticker=result["prediction"]["target_ticker"],
                input_tickers=result["prediction"]["input_tickers"],
                predicted_return_pct=result["prediction"]["predicted_return_pct"],
                confidence_interval=ConfidenceInterval(
                    lower=result["prediction"]["confidence_interval"]["lower"],
                    upper=result["prediction"]["confidence_interval"]["upper"],
                    confidence_level=result["prediction"]["confidence_interval"][
                        "confidence_level"
                    ],
                ),
                forecast_horizon_days=result["prediction"]["forecast_horizon_days"],
                prediction_timestamp=result["prediction"]["prediction_timestamp"],
                features_used=result["prediction"]["features_used"],
            ),
            model_info=ModelInfo(
                model_id=result["model_id"],
                symbol=result["prediction"]["target_ticker"],
                trained_at=result["model_info"]["trained_at"],
                data_rows=result["model_info"]["data_rows"],
                config=result["model_info"]["config"],
                mlflow_run_id=result.get("mlflow_run_id"),
            ),
        )

        elapsed_time = time.time() - start_time
        logger.info(
            "multivariate_train_predict_completed",
            target_ticker=request.target_ticker,
            predicted_return=result["prediction"]["predicted_return_pct"],
            total_time=elapsed_time,
        )

        return response

    except ValidationError as e:
        logger.error("validation_failed", error=str(e))
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Validation failed: {str(e)}",
                "error_type": "ValidationError",
            },
        )
    except ModelTrainingError as e:
        logger.error("training_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Model training failed: {str(e)}",
                "error_type": "ModelTrainingError",
            },
        )
    except Exception as e:
        logger.error(
            "unexpected_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Unexpected error: {str(e)}",
                "error_type": type(e).__name__,
            },
        )


@router.post("/predict", response_model=MultivariatePredictionResponse)
async def predict_multivariate(
    request: MultivariatePredictRequest,
    use_case: PredictMultivariateUseCase = Depends(get_multivariate_use_case),
) -> MultivariatePredictionResponse:
    """
    Generate prediction using pre-trained multivariate model from MLflow Registry.

    This endpoint:
    1. Loads pre-trained model from MLflow Registry
    2. Fetches latest historical data for input tickers
    3. Calculates same features used during training
    4. Generates prediction with Monte Carlo Dropout uncertainty
    5. Returns prediction without retraining

    **Note:** Model must be previously trained via `/multivariate/train-predict` endpoint
    and registered in MLflow Registry.

    **Example Request:**
    ```json
    {
        "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
        "target_ticker": "NVDA",
        "model_version": "latest",
        "lookback": 60,
        "forecast_horizon": 5,
        "confidence_level": 0.95
    }
    ```

    Args:
        request: Multivariate prediction request
        use_case: Injected use case dependency

    Returns:
        MultivariatePredictionResponse with prediction and model info

    Raises:
        HTTPException 404: If model not found in registry
        HTTPException 500: If prediction fails
    """
    logger.info(
        "multivariate_predict_request_received",
        input_tickers=request.input_tickers,
        target_ticker=request.target_ticker,
        model_version=request.model_version,
    )

    # TODO: Implement model loading from MLflow Registry
    # For now, return error message
    raise HTTPException(
        status_code=501,
        detail={
            "message": "Predict-only endpoint not yet implemented. "
            "Please use /multivariate/train-predict endpoint to train and predict in one call.",
            "error_type": "NotImplementedError",
        },
    )
