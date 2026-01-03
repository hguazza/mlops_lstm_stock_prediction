"""FastAPI dependency injection."""

from functools import lru_cache
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from src.application.services import (
    create_data_service,
    create_prediction_service_with_mlflow,
)
from src.application.services.data_service import DataService
from src.application.services.prediction_service import PredictionService
from src.application.use_cases.fetch_data import FetchStockDataUseCase
from src.application.use_cases.predict_stock import PredictStockPriceUseCase
from src.infrastructure.database.connection import get_async_session
from src.infrastructure.database.repositories import (
    RefreshTokenRepository,
    UserRepository,
)
from src.infrastructure.mlflow.mlflow_service import MLflowService


@lru_cache()
def get_data_service() -> DataService:
    """
    Get DataService instance (singleton).

    Returns:
        DataService instance
    """
    return create_data_service()


@lru_cache()
def get_prediction_service_with_mlflow() -> PredictionService:
    """
    Get PredictionService with MLflow tracking (singleton).

    Returns:
        PredictionService with MLflow enabled
    """
    return create_prediction_service_with_mlflow()


def get_mlflow_service() -> MLflowService:
    """
    Get MLflowService from prediction service.

    Returns:
        MLflowService instance

    Raises:
        RuntimeError: If MLflow is not configured
    """
    prediction_service = get_prediction_service_with_mlflow()
    if prediction_service.mlflow_service is None:
        raise RuntimeError("MLflow service not configured")
    return prediction_service.mlflow_service


def get_fetch_stock_data_use_case() -> FetchStockDataUseCase:
    """
    Get FetchStockDataUseCase with injected dependencies.

    Returns:
        FetchStockDataUseCase instance
    """
    data_service = get_data_service()
    return FetchStockDataUseCase(data_service=data_service)


def get_predict_stock_price_use_case() -> PredictStockPriceUseCase:
    """
    Get PredictStockPriceUseCase with injected dependencies.

    Returns:
        PredictStockPriceUseCase instance
    """
    data_service = get_data_service()
    prediction_service = get_prediction_service_with_mlflow()
    return PredictStockPriceUseCase(
        data_service=data_service,
        prediction_service=prediction_service,
    )


# Database session dependency (re-exported from connection module)
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session dependency.

    Yields:
        AsyncSession: Database session
    """
    async for session in get_async_session():
        yield session


def get_user_repository(
    session: AsyncSession,
) -> UserRepository:
    """
    Get UserRepository with injected session.

    Args:
        session: Database session

    Returns:
        UserRepository instance
    """
    return UserRepository(session)


def get_refresh_token_repository(
    session: AsyncSession,
) -> RefreshTokenRepository:
    """
    Get RefreshTokenRepository with injected session.

    Args:
        session: Database session

    Returns:
        RefreshTokenRepository instance
    """
    return RefreshTokenRepository(session)
