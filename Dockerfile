# Multi-stage Dockerfile for Stock Prediction API

# Stage 1: Builder - Install dependencies
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies to /app/.venv
RUN uv sync --no-dev

# Stage 2: Runtime - Minimal production image
FROM python:3.12-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Create data directories for MLflow with proper permissions
RUN mkdir -p /data && chown appuser:appuser /data
RUN mkdir -p /app/mlruns_artifacts && chown appuser:appuser /app/mlruns_artifacts

# Copy and make entrypoint executable
COPY scripts/docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Note: Running as root for Docker volume permissions
# USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///data/mlflow.db

# Set entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]

# Run API with uvicorn
CMD ["uvicorn", "src.presentation.main:app", "--host", "0.0.0.0", "--port", "8000"]
