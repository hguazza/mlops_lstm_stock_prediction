"""Health check router."""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends

from src.domain.auth.security import get_current_active_user
from src.infrastructure.database.models import User
from src.presentation.api.dependencies import get_mlflow_service
from src.presentation.schemas.common import HealthResponse, StatusEnum

router = APIRouter(tags=["health"])
logger = structlog.get_logger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check(
    current_user: User = Depends(get_current_active_user),
) -> HealthResponse:
    """
    Health check endpoint.

    Checks the status of the API and its dependencies.

    Returns:
        HealthResponse with status and dependency health
    """
    dependencies = {}

    # Check MLflow
    try:
        mlflow_service = get_mlflow_service()
        if mlflow_service and mlflow_service._setup_complete:
            dependencies["mlflow"] = "connected"
        else:
            dependencies["mlflow"] = "not_configured"
    except Exception as e:
        logger.warning("mlflow_health_check_failed", error=str(e))
        dependencies["mlflow"] = "error"

    # Check yfinance (simple test)
    try:
        import yfinance as yf

        # Try to create a ticker instance (doesn't make API call)
        _ = yf.Ticker("AAPL")
        dependencies["yfinance"] = "accessible"
    except Exception as e:
        logger.warning("yfinance_health_check_failed", error=str(e))
        dependencies["yfinance"] = "error"

    # Determine overall status
    all_healthy = all(v in ["connected", "accessible"] for v in dependencies.values())
    overall_status = StatusEnum.HEALTHY if all_healthy else StatusEnum.UNHEALTHY

    logger.info(
        "health_check_completed", status=overall_status, dependencies=dependencies
    )

    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        dependencies=dependencies,
    )
