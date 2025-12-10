#!/usr/bin/env python
"""MLflow initialization script.

Sets up MLflow tracking database, creates default experiment,
and verifies configuration.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.infrastructure.mlflow import MLflowConfig
from src.infrastructure.mlflow.mlflow_service import MLflowService


def init_mlflow():
    """Initialize MLflow tracking and experiment."""
    print("=" * 60)
    print("MLflow Initialization Script")
    print("=" * 60)

    # Load configuration
    app_config = get_config()
    mlflow_config = MLflowConfig(
        tracking_uri=app_config.mlflow.tracking_uri,
        experiment_name=app_config.mlflow.experiment_name,
        artifact_location=app_config.mlflow.artifact_location,
        enable_autolog=app_config.mlflow.enable_autolog,
    )

    print("\nConfiguration:")
    print(f"  Tracking URI: {mlflow_config.tracking_uri}")
    print(f"  Experiment Name: {mlflow_config.experiment_name}")
    print(f"  Artifact Location: {mlflow_config.artifact_location}")
    print(f"  Autologging: {mlflow_config.enable_autolog}")

    # Create MLflow service
    print("\n[1/3] Creating MLflow service...")
    try:
        mlflow_service = MLflowService(config=mlflow_config)
        print("  [OK] MLflow service created")
    except Exception as e:
        print(f"  [ERROR] Failed to create MLflow service: {e}")
        return 1

    # Setup tracking
    print("\n[2/3] Setting up MLflow tracking...")
    try:
        mlflow_service.setup_tracking()
        print("  [OK] MLflow tracking configured")
        print(f"  [OK] Experiment '{mlflow_config.experiment_name}' ready")
        print(f"  [OK] Experiment ID: {mlflow_service._experiment_id}")
    except Exception as e:
        print(f"  [ERROR] Failed to setup tracking: {e}")
        return 1

    # Verify artifact directory
    print("\n[3/3] Verifying artifact directory...")
    artifact_path = Path(mlflow_config.artifact_location)
    if artifact_path.exists():
        print(f"  [OK] Artifact directory exists: {artifact_path}")
    else:
        print("  [INFO] Artifact directory will be created on first use")

    # Display database location
    if mlflow_config.tracking_uri.startswith("sqlite:///"):
        db_path = mlflow_config.tracking_uri.replace("sqlite:///", "")
        db_full_path = Path(db_path).resolve()
        print(f"\n[INFO] SQLite database: {db_full_path}")

    # Display UI command
    print("\n" + "=" * 60)
    print("Initialization Complete!")
    print("=" * 60)
    print("\nTo view experiments in the MLflow UI, run:")
    print(f"  mlflow ui --backend-store-uri {mlflow_config.tracking_uri}")
    print("\nThen open: http://localhost:5000")
    print("\nTo use MLflow in your code:")
    print(
        "  from src.application.services import create_prediction_service_with_mlflow"
    )
    print("  prediction_service = create_prediction_service_with_mlflow()")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(init_mlflow())
