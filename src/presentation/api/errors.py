"""Custom exceptions and error handlers for API."""

from datetime import datetime
from typing import Any, Dict

import structlog
from fastapi import Request, status
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "APIError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Dict[str, Any] = None,
    ):
        """
        Initialize API error.

        Args:
            message: Error message
            error_code: Error code/type
            status_code: HTTP status code
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(APIError):
    """Raised when request validation fails."""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """Initialize validation error."""
        super().__init__(
            message=message,
            error_code="ValidationError",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )


class ModelNotFoundError(APIError):
    """Raised when model is not found in registry."""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """Initialize model not found error."""
        super().__init__(
            message=message,
            error_code="ModelNotFoundError",
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class TrainingError(APIError):
    """Raised when model training fails."""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """Initialize training error."""
        super().__init__(
            message=message,
            error_code="TrainingError",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class PredictionError(APIError):
    """Raised when prediction fails."""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """Initialize prediction error."""
        super().__init__(
            message=message,
            error_code="PredictionError",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class DataFetchError(APIError):
    """Raised when data fetching fails."""

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """Initialize data fetch error."""
        super().__init__(
            message=message,
            error_code="DataFetchError",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """
    Handle APIError exceptions.

    Args:
        request: FastAPI request
        exc: APIError exception

    Returns:
        JSONResponse with error details
    """
    logger.error(
        "api_error",
        error_code=exc.error_code,
        message=exc.message,
        status_code=exc.status_code,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions.

    Args:
        request: FastAPI request
        exc: Exception

    Returns:
        JSONResponse with generic error
    """
    logger.error(
        "unhandled_exception",
        error_type=type(exc).__name__,
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"error_type": type(exc).__name__},
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


async def validation_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Handle Pydantic ValidationError.

    Args:
        request: FastAPI request
        exc: ValidationError from Pydantic

    Returns:
        JSONResponse with validation details
    """
    logger.warning(
        "validation_error",
        path=request.url.path,
        method=request.method,
    )

    # Extract validation errors
    details = {}
    if hasattr(exc, "errors"):
        details["validation_errors"] = exc.errors()

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
