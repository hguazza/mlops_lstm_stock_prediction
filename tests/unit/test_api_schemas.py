"""Unit tests for API Pydantic schemas."""

import pytest
from pydantic import ValidationError

from src.presentation.schemas.common import ErrorResponse, HealthResponse, StatusEnum
from src.presentation.schemas.requests import (
    ConfigOverride,
    PredictRequest,
    TrainRequest,
)
from src.presentation.schemas.responses import (
    ConfidenceInterval,
    ModelInfo,
    PredictionDetails,
    TrainingMetrics,
)


class TestCommonSchemas:
    """Test common schemas."""

    def test_status_enum_values(self):
        """Test StatusEnum has expected values."""
        assert StatusEnum.SUCCESS == "success"
        assert StatusEnum.ERROR == "error"
        assert StatusEnum.HEALTHY == "healthy"
        assert StatusEnum.UNHEALTHY == "unhealthy"

    def test_health_response_creation(self):
        """Test HealthResponse creation."""
        response = HealthResponse(
            status=StatusEnum.HEALTHY,
            dependencies={"mlflow": "connected", "yfinance": "accessible"},
        )
        assert response.status == StatusEnum.HEALTHY
        assert response.version == "0.1.0"
        assert "mlflow" in response.dependencies

    def test_error_response_creation(self):
        """Test ErrorResponse creation."""
        response = ErrorResponse(
            error="TestError",
            message="Test error message",
            details={"key": "value"},
        )
        assert response.error == "TestError"
        assert response.message == "Test error message"
        assert response.details == {"key": "value"}


class TestRequestSchemas:
    """Test request schemas."""

    def test_train_request_valid(self):
        """Test valid TrainRequest."""
        request = TrainRequest(
            symbol="AAPL",
            period="1y",
        )
        assert request.symbol == "AAPL"
        assert request.period == "1y"
        assert request.config is None

    def test_train_request_with_config(self):
        """Test TrainRequest with config override."""
        config = ConfigOverride(
            hidden_size=64,
            num_layers=3,
            epochs=50,
        )
        request = TrainRequest(
            symbol="GOOGL",
            period="2y",
            config=config,
        )
        assert request.config.hidden_size == 64
        assert request.config.num_layers == 3
        assert request.config.epochs == 50

    def test_train_request_symbol_uppercase(self):
        """Test symbol is converted to uppercase."""
        request = TrainRequest(symbol="aapl", period="1y")
        assert request.symbol == "AAPL"

    def test_train_request_invalid_period(self):
        """Test invalid period raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TrainRequest(symbol="AAPL", period="invalid")
        assert "Invalid period" in str(exc_info.value)

    def test_predict_request_valid(self):
        """Test valid PredictRequest."""
        request = PredictRequest(symbol="AAPL")
        assert request.symbol == "AAPL"
        assert request.model_version == "latest"

    def test_predict_request_with_version(self):
        """Test PredictRequest with specific version."""
        request = PredictRequest(symbol="MSFT", model_version="3")
        assert request.model_version == "3"

    def test_config_override_validation(self):
        """Test ConfigOverride field validation."""
        # Valid config
        config = ConfigOverride(
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            epochs=100,
        )
        assert config.hidden_size == 128

        # Invalid hidden_size (too small)
        with pytest.raises(ValidationError):
            ConfigOverride(hidden_size=8)

        # Invalid num_layers (too many)
        with pytest.raises(ValidationError):
            ConfigOverride(num_layers=10)


class TestResponseSchemas:
    """Test response schemas."""

    def test_training_metrics(self):
        """Test TrainingMetrics schema."""
        metrics = TrainingMetrics(
            best_val_loss=0.023,
            final_train_loss=0.018,
            final_val_loss=0.025,
            epochs_trained=45,
            training_time_seconds=120.5,
        )
        assert metrics.best_val_loss == 0.023
        assert metrics.epochs_trained == 45

    def test_confidence_interval(self):
        """Test ConfidenceInterval schema."""
        ci = ConfidenceInterval(
            lower=175.0,
            upper=182.0,
            confidence_level=0.95,
        )
        assert ci.lower == 175.0
        assert ci.upper == 182.0
        assert ci.confidence_level == 0.95

    def test_prediction_details(self):
        """Test PredictionDetails schema."""

        prediction = PredictionDetails(
            symbol="AAPL",
            predicted_price=178.45,
            confidence_interval=ConfidenceInterval(
                lower=175.0,
                upper=182.0,
            ),
            prediction_date="2024-12-11",
            current_price=176.80,
        )
        assert prediction.symbol == "AAPL"
        assert prediction.predicted_price == 178.45
        assert prediction.current_price == 176.80

    def test_model_info(self):
        """Test ModelInfo schema."""
        from datetime import datetime

        info = ModelInfo(
            model_id="stock_predictor_AAPL:3",
            symbol="AAPL",
            trained_at=datetime.utcnow(),
            data_rows=252,
            config={"hidden_size": 50, "num_layers": 2},
        )
        assert info.model_id == "stock_predictor_AAPL:3"
        assert info.symbol == "AAPL"
        assert info.data_rows == 252
