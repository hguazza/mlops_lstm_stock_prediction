"""FastAPI application main module."""

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError as PydanticValidationError

from src.presentation.api.errors import (
    APIError,
    api_error_handler,
    general_exception_handler,
    validation_exception_handler,
)
from src.presentation.api.middleware.metrics import (
    PrometheusMiddleware,
    metrics_endpoint,
)

from src.presentation.api.routers import health, models, multivariate, predict, train

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Price Prediction API",
    description="LSTM-based stock price prediction API with MLflow tracking",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("application_starting", version="0.1.0")

    # Initialize MLflow if needed
    try:
        from src.config import get_config

        config = get_config()
        logger.info(
            "mlflow_configured",
            tracking_uri=config.mlflow.tracking_uri,
            experiment_name=config.mlflow.experiment_name,
        )
    except Exception as e:
        logger.warning("mlflow_initialization_warning", error=str(e))

    logger.info("application_started")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("application_shutting_down")


# Exception handlers
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(PydanticValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    logger.info(
        "request_received",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else None,
    )

    response = await call_next(request)

    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
    )

    return response


# Include routers with /api/v1 prefix
app.include_router(health.router, prefix="/api/v1")
app.include_router(train.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")
app.include_router(multivariate.router, prefix="/api/v1")
app.include_router(models.router, prefix="/api/v1")


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return await metrics_endpoint()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Stock Price Prediction API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "metrics": "/metrics",
    }
