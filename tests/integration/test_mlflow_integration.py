"""Integration tests for MLflow tracking with real training."""

import tempfile
from pathlib import Path

import pytest

from src.application.services import create_prediction_service_with_mlflow
from src.infrastructure.mlflow import MLflowConfig


@pytest.mark.integration
@pytest.mark.slow
def test_mlflow_end_to_end_tracking(small_stock_data):
    """Test complete MLflow tracking workflow with actual training."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create MLflow config with temporary directory
        mlflow_config = MLflowConfig(
            tracking_uri=f"sqlite:///{tmpdir}/test_mlflow.db",
            experiment_name="test_experiment",
            artifact_location=f"{tmpdir}/artifacts",
            enable_autolog=False,  # Disable for faster testing
        )

        # Create prediction service with MLflow
        prediction_service = create_prediction_service_with_mlflow(mlflow_config)

        # Verify MLflow is configured
        assert prediction_service.mlflow_service is not None
        assert prediction_service.mlflow_service._setup_complete is True

        # Train model with MLflow tracking
        history = prediction_service.train_model(
            data=small_stock_data,
            target_column="Close",
            verbose=False,
            symbol="TEST",
        )

        # Verify training completed
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0

        # Verify MLflow database was created
        db_path = Path(tmpdir) / "test_mlflow.db"
        assert db_path.exists()

        # Verify artifacts directory was created
        artifact_path = Path(tmpdir) / "artifacts"
        assert artifact_path.exists()


@pytest.mark.integration
def test_mlflow_service_without_training(small_stock_data):
    """Test that prediction service works without MLflow."""
    from src.application.services import create_prediction_service

    # Create service without MLflow
    prediction_service = create_prediction_service()

    # Verify no MLflow
    assert prediction_service.mlflow_service is None

    # Train should still work
    history = prediction_service.train_model(
        data=small_stock_data,
        target_column="Close",
        verbose=False,
    )

    # Verify training completed
    assert "train_loss" in history
    assert len(history["train_loss"]) > 0
