#!/bin/bash
set -e

echo "Starting Stock Prediction API..."

# Ensure data directory exists and has correct permissions
mkdir -p /data
chmod 755 /data

# Initialize MLflow if database doesn't exist
if [ ! -f "/data/mlflow.db" ]; then
    echo "Initializing MLflow for the first time..."
    echo "MLflow tracking URI: $MLFLOW_TRACKING_URI"
    echo "Artifact location: $MLFLOW_ARTIFACT_LOCATION"

    # Create artifact directory
    mkdir -p /data/mlruns_artifacts

    # Initialize MLflow
    python scripts/init_mlflow.py

    if [ -f "/data/mlflow.db" ]; then
        echo "MLflow initialized successfully!"
        ls -la /data/
    else
        echo "WARNING: MLflow initialization may have failed"
    fi
else
    echo "MLflow database already exists, skipping initialization."
    ls -la /data/
fi

# Execute the main command
exec "$@"
