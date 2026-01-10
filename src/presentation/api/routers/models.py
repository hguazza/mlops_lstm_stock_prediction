"""Models router - Model registry and information endpoints."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import mlflow
import structlog
from fastapi import APIRouter, Depends, Path

from src.presentation.api.dependencies import get_mlflow_service
from src.presentation.api.errors import ModelNotFoundError
from src.presentation.schemas.responses import (
    ModelInfo,
    ModelInfoResponse,
    ModelMetrics,
    ModelsListResponse,
    ModelSummary,
)

router = APIRouter(prefix="/models", tags=["models"])
logger = structlog.get_logger(__name__)

_STOCK_MODEL_PREFIX = "stock_predictor_"
_MULTIVARIATE_MODEL_PREFIX = "multivariate_predictor_"


def _extract_symbol_from_model_name(model_name: str) -> str:
    """
    Extract the display symbol from a registered model name.

    Supports both:
    - stock_predictor_{SYMBOL}
    - multivariate_predictor_{SYMBOL}
    """
    if model_name.startswith(_STOCK_MODEL_PREFIX):
        return model_name.removeprefix(_STOCK_MODEL_PREFIX)
    if model_name.startswith(_MULTIVARIATE_MODEL_PREFIX):
        return model_name.removeprefix(_MULTIVARIATE_MODEL_PREFIX)
    return model_name


def _resolve_model_name_for_symbol(
    client: mlflow.MlflowClient, symbol: str
) -> Tuple[Optional[str], List]:
    """
    Resolve the actual registered model name + versions list for a symbol.

    Tries both naming conventions (univariate first, then multivariate).
    Returns (model_name, versions). If not found, (None, []).
    """
    candidates = [
        f"{_STOCK_MODEL_PREFIX}{symbol}",
        f"{_MULTIVARIATE_MODEL_PREFIX}{symbol}",
    ]
    for model_name in candidates:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            return model_name, versions
    return None, []


@router.get("", response_model=ModelsListResponse)
async def list_models(
    mlflow_service=Depends(get_mlflow_service),
) -> ModelsListResponse:
    """
    List all registered models from MLflow Registry.

    Returns:
        ModelsListResponse with list of all registered models and their metadata

    Raises:
        ModelNotFoundError: If no models found or MLflow error
    """
    logger.info("list_models_request_received")

    try:
        client = mlflow.MlflowClient()

        # Get all registered models
        registered_models = client.search_registered_models()

        models: List[ModelSummary] = []

        for rm in registered_models:
            model_name = rm.name
            symbol = _extract_symbol_from_model_name(model_name)

            # Get all versions
            versions = client.search_model_versions(f"name='{model_name}'")

            if not versions:
                continue

            # Sort versions by version number
            sorted_versions = sorted(
                versions, key=lambda v: int(v.version), reverse=True
            )

            latest_version = str(sorted_versions[0].version)

            # Find production version
            production_version = None
            for v in sorted_versions:
                if v.current_stage == "Production":
                    production_version = str(v.version)
                    break

            # Get last trained timestamp
            last_trained = None
            if sorted_versions[0].creation_timestamp:
                last_trained = datetime.fromtimestamp(
                    sorted_versions[0].creation_timestamp / 1000
                )

            models.append(
                ModelSummary(
                    symbol=symbol,
                    versions=len(versions),
                    latest_version=latest_version,
                    production_version=production_version,
                    last_trained=last_trained,
                )
            )

        logger.info("list_models_completed", total_models=len(models))

        return ModelsListResponse(
            models=models,
            total_models=len(models),
        )

    except Exception as e:
        logger.error(
            "list_models_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise ModelNotFoundError(
            message="Failed to retrieve models from registry",
            details={"error_type": type(e).__name__},
        )


@router.get("/{symbol}/latest", response_model=ModelInfoResponse)
async def get_latest_model_info(
    symbol: str = Path(..., description="Stock symbol"),
    mlflow_service=Depends(get_mlflow_service),
) -> ModelInfoResponse:
    """
    Get information about the latest model for a specific symbol.

    Args:
        symbol: Stock symbol (e.g., AAPL)
        mlflow_service: Injected MLflow service

    Returns:
        ModelInfoResponse with detailed model information

    Raises:
        ModelNotFoundError: If model not found
    """
    logger.info("get_model_info_request_received", symbol=symbol)

    symbol = symbol.upper()

    try:
        client = mlflow.MlflowClient()

        model_name, versions = _resolve_model_name_for_symbol(client, symbol)
        if not model_name or not versions:
            raise ModelNotFoundError(
                message=f"No model found for symbol {symbol}",
                details={
                    "symbol": symbol,
                    "tried_model_names": [
                        f"{_STOCK_MODEL_PREFIX}{symbol}",
                        f"{_MULTIVARIATE_MODEL_PREFIX}{symbol}",
                    ],
                },
            )

        # Sort and get latest
        sorted_versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
        latest = sorted_versions[0]

        # Get run details for metrics
        run = mlflow.get_run(latest.run_id)

        # Extract metrics
        metrics = None
        if run.data.metrics:
            metrics = ModelMetrics(
                best_val_loss=run.data.metrics.get("best_val_loss"),
                final_train_loss=run.data.metrics.get("final_train_loss"),
                final_val_loss=run.data.metrics.get("final_val_loss"),
                epochs_trained=int(run.data.metrics.get("epochs_trained", 0))
                if run.data.metrics.get("epochs_trained")
                else None,
            )

        # Extract config from params
        config: Dict = {}
        if run.data.params:
            for key, value in run.data.params.items():
                if key.startswith("config_"):
                    config_key = key.replace("config_", "")
                    try:
                        # Try to convert to appropriate type
                        if "." in value:
                            config[config_key] = float(value)
                        else:
                            config[config_key] = int(value)
                    except ValueError:
                        config[config_key] = value

        # Build model info
        trained_at = datetime.fromtimestamp(latest.creation_timestamp / 1000)

        model_info = ModelInfo(
            model_id=f"{model_name}:{latest.version}",
            symbol=symbol,
            trained_at=trained_at,
            data_rows=int(run.data.params.get("data_rows", 0)),
            config=config,
            mlflow_run_id=latest.run_id,
            model_stage=latest.current_stage,
        )

        logger.info(
            "get_model_info_completed",
            symbol=symbol,
            version=latest.version,
            stage=latest.current_stage,
        )

        return ModelInfoResponse(
            model_info=model_info,
            metrics=metrics,
        )

    except mlflow.exceptions.MlflowException as e:
        logger.error(
            "get_model_info_failed",
            symbol=symbol,
            error=str(e),
        )
        raise ModelNotFoundError(
            message=f"Model not found for symbol {symbol}",
            details={"symbol": symbol, "error": str(e)},
        )
    except Exception as e:
        logger.error(
            "get_model_info_failed",
            symbol=symbol,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise ModelNotFoundError(
            message=f"Failed to retrieve model info for {symbol}",
            details={"symbol": symbol, "error_type": type(e).__name__},
        )
