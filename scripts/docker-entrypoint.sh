#!/bin/bash
set -e

echo "Starting Stock Prediction API..."

# Ensure data directory exists and has correct permissions
mkdir -p /data
chmod 755 /data

# Function to initialize authentication database
init_auth_db() {
    echo "Initializing authentication database..."
    echo "Database URL: $DATABASE_URL"

    python -c "
import asyncio
from src.infrastructure.database.models import Base
from src.infrastructure.database.connection import engine

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print('✓ Authentication tables created successfully!')

try:
    asyncio.run(create_tables())
except Exception as e:
    print(f'⚠ Warning: Could not create auth tables: {e}')
"
}

# Function to initialize MLflow database
init_mlflow() {
    echo "Initializing MLflow database..."
    echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
    echo "Artifact location: $MLFLOW_ARTIFACT_LOCATION"

    # Create artifact directory
    mkdir -p /data/mlruns_artifacts

    # Remove old database if it exists (to avoid migration conflicts)
    if [ -f "/data/mlflow.db" ]; then
        echo "Removing old MLflow database to avoid migration conflicts..."
        rm -f /data/mlflow.db /data/mlflow.db-shm /data/mlflow.db-wal
    fi

    # Initialize MLflow
    python scripts/init_mlflow.py

    if [ -f "/data/mlflow.db" ]; then
        echo "MLflow initialized successfully!"
        ls -la /data/
    else
        echo "ERROR: MLflow initialization failed"
        exit 1
    fi
}

# Check if MLflow database exists
if [ -f "/data/mlflow.db" ]; then
    echo "MLflow database found. Checking database health..."

    # Try to query the database to check if it's healthy
    if ! python -c "import mlflow; mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI'); mlflow.search_experiments()" 2>/dev/null; then
        echo "WARNING: MLflow database appears to be corrupted or has migration issues."
        echo "Reinitializing database..."
        init_mlflow
    else
        echo "MLflow database is healthy."
        ls -la /data/
    fi
else
    echo "No MLflow database found. Initializing for the first time..."
    init_mlflow
fi

# Initialize authentication database
echo "Checking authentication database..."
init_auth_db

# Execute the main command
exec "$@"
