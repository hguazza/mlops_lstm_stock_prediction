"""Unit tests for MLflowService."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch.nn as nn

from src.infrastructure.mlflow import MLflowConfig
from src.infrastructure.mlflow.mlflow_service import MLflowService, MLflowServiceError


@pytest.mark.unit
class TestMLflowServiceInit:
    """Test MLflowService initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        service = MLflowService()
        assert service.config is not None
        assert service.config.tracking_uri == "sqlite:///mlflow.db"
        assert service._experiment_id is None
        assert service._active_run is None

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = MLflowConfig(
            tracking_uri="sqlite:///custom.db",
            experiment_name="test_experiment",
        )
        service = MLflowService(config=config)
        assert service.config == config
        assert service.config.experiment_name == "test_experiment"


@pytest.mark.unit
class TestSetupTracking:
    """Test MLflow tracking setup."""

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_setup_tracking_creates_experiment(self, mock_mlflow):
        """Test that setup creates new experiment if it doesn't exist."""
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "exp123"

        service = MLflowService()
        service.setup_tracking()

        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.get_experiment_by_name.assert_called_once()
        mock_mlflow.create_experiment.assert_called_once()
        assert service._experiment_id == "exp123"
        assert service._setup_complete is True

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_setup_tracking_uses_existing_experiment(self, mock_mlflow):
        """Test that setup uses existing experiment."""
        mock_experiment = Mock()
        mock_experiment.experiment_id = "exp456"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        service = MLflowService()
        service.setup_tracking()

        mock_mlflow.create_experiment.assert_not_called()
        assert service._experiment_id == "exp456"

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_setup_tracking_failure_raises_error(self, mock_mlflow):
        """Test that setup failure raises MLflowServiceError."""
        mock_mlflow.set_tracking_uri.side_effect = Exception("Connection error")

        service = MLflowService()
        with pytest.raises(MLflowServiceError, match="Failed to setup MLflow"):
            service.setup_tracking()


@pytest.mark.unit
class TestStartRun:
    """Test starting MLflow runs."""

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_start_run_without_setup(self, mock_mlflow):
        """Test that start_run calls setup if not done."""
        mock_experiment = Mock()
        mock_experiment.experiment_id = "exp123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_run = Mock()
        mock_run.info.run_id = "run123"
        mock_mlflow.start_run.return_value = mock_run

        service = MLflowService()
        service.start_run(run_name="test_run")

        assert service._setup_complete is True
        mock_mlflow.start_run.assert_called_once()

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_start_run_with_tags(self, mock_mlflow):
        """Test starting run with tags."""
        service = MLflowService()
        service._setup_complete = True
        service._experiment_id = "exp123"

        mock_run = Mock()
        mock_run.info.run_id = "run123"
        mock_mlflow.start_run.return_value = mock_run

        tags = {"model": "LSTM", "version": "1.0"}
        service.start_run(run_name="test_run", tags=tags)

        mock_mlflow.set_tags.assert_called_once_with(tags)


@pytest.mark.unit
class TestLogParams:
    """Test parameter logging."""

    def test_log_params_without_active_run(self):
        """Test that log_params raises error without active run."""
        service = MLflowService()
        with pytest.raises(MLflowServiceError, match="No active run"):
            service.log_params({"param1": "value1"})

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_log_params_success(self, mock_mlflow):
        """Test successful parameter logging."""
        service = MLflowService()
        service._active_run = Mock()

        params = {"hidden_size": 50, "num_layers": 2}
        service.log_params(params)

        mock_mlflow.log_params.assert_called_once_with(params)


@pytest.mark.unit
class TestLogMetrics:
    """Test metric logging."""

    def test_log_metrics_without_active_run(self):
        """Test that log_metrics raises error without active run."""
        service = MLflowService()
        with pytest.raises(MLflowServiceError, match="No active run"):
            service.log_metrics({"loss": 0.5})

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_log_metrics_success(self, mock_mlflow):
        """Test successful metric logging."""
        service = MLflowService()
        service._active_run = Mock()

        metrics = {"train_loss": 0.3, "val_loss": 0.4}
        service.log_metrics(metrics, step=10)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=10)


@pytest.mark.unit
class TestInferSignature:
    """Test model signature inference."""

    @patch("src.infrastructure.mlflow.mlflow_service.infer_signature")
    def test_infer_signature(self, mock_infer):
        """Test signature inference."""
        mock_signature = Mock()
        mock_infer.return_value = mock_signature

        service = MLflowService()
        input_data = np.array([[1, 2, 3]])
        output_data = np.array([[0.5]])

        signature = service.infer_signature(input_data, output_data)

        mock_infer.assert_called_once_with(input_data, output_data)
        assert signature == mock_signature


@pytest.mark.unit
class TestLogModel:
    """Test model logging."""

    def test_log_model_without_active_run(self):
        """Test that log_model raises error without active run."""
        service = MLflowService()
        model = Mock(spec=nn.Module)

        with pytest.raises(MLflowServiceError, match="No active run"):
            service.log_model(model)

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow.pytorch")
    def test_log_model_success(self, mock_pytorch):
        """Test successful model logging."""
        service = MLflowService()
        service._active_run = Mock()

        model = Mock(spec=nn.Module)
        mock_model_info = Mock()
        mock_model_info.model_uri = "runs:/123/model"
        mock_pytorch.log_model.return_value = mock_model_info

        model_uri = service.log_model(
            model=model,
            artifact_path="model",
            registered_model_name="test_model",
        )

        assert model_uri == "runs:/123/model"
        mock_pytorch.log_model.assert_called_once()


@pytest.mark.unit
class TestEndRun:
    """Test ending MLflow runs."""

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_end_run_success(self, mock_mlflow):
        """Test successful run ending."""
        service = MLflowService()
        mock_run = Mock()
        mock_run.info.run_id = "run123"
        service._active_run = mock_run

        service.end_run(status="FINISHED")

        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")
        assert service._active_run is None

    def test_end_run_without_active_run(self):
        """Test ending run when no active run."""
        service = MLflowService()
        service.end_run()  # Should not raise error


@pytest.mark.unit
class TestContextManager:
    """Test context manager behavior."""

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_context_manager_success(self, mock_mlflow):
        """Test context manager with successful execution."""
        service = MLflowService()
        service._setup_complete = True
        service._experiment_id = "exp123"

        mock_run = Mock()
        mock_run.info.run_id = "run123"
        mock_mlflow.start_run.return_value = mock_run

        with service:
            assert service._active_run is not None

        mock_mlflow.end_run.assert_called_once_with(status="FINISHED")

    @patch("src.infrastructure.mlflow.mlflow_service.mlflow")
    def test_context_manager_with_exception(self, mock_mlflow):
        """Test context manager with exception."""
        service = MLflowService()
        service._setup_complete = True
        service._experiment_id = "exp123"

        mock_run = Mock()
        mock_run.info.run_id = "run123"
        mock_mlflow.start_run.return_value = mock_run

        with pytest.raises(ValueError):
            with service:
                raise ValueError("Test error")

        mock_mlflow.end_run.assert_called_once_with(status="FAILED")
