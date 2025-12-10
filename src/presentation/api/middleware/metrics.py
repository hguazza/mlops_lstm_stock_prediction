"""Prometheus metrics middleware and endpoint."""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

# Define metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

predictions_total = Counter(
    "predictions_total",
    "Total predictions made",
    ["symbol", "status"],
)

training_total = Counter(
    "training_total",
    "Total training requests",
    ["symbol", "status"],
)

model_load_duration_seconds = Histogram(
    "model_load_duration_seconds",
    "Model loading duration in seconds",
    ["symbol"],
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics.

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint

        Returns:
            Response from endpoint
        """
        # Skip metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        # Start timer
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Extract endpoint path (remove query params)
        endpoint = request.url.path

        # Record metrics
        http_requests_total.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()

        http_request_duration_seconds.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)

        return response


async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.

    Returns:
        Response with metrics in Prometheus format
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


def record_prediction_metric(symbol: str, success: bool) -> None:
    """
    Record a prediction metric.

    Args:
        symbol: Stock symbol
        success: Whether prediction was successful
    """
    predictions_total.labels(
        symbol=symbol,
        status="success" if success else "error",
    ).inc()


def record_training_metric(symbol: str, success: bool) -> None:
    """
    Record a training metric.

    Args:
        symbol: Stock symbol
        success: Whether training was successful
    """
    training_total.labels(
        symbol=symbol,
        status="success" if success else "error",
    ).inc()


def record_model_load_duration(symbol: str, duration: float) -> None:
    """
    Record model loading duration.

    Args:
        symbol: Stock symbol
        duration: Loading duration in seconds
    """
    model_load_duration_seconds.labels(symbol=symbol).observe(duration)
