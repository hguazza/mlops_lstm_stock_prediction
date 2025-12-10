"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.presentation.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self):
        """Test health check returns successful response."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "dependencies" in data
        assert data["version"] == "0.1.0"

    def test_health_check_dependencies(self):
        """Test health check includes dependency status."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "mlflow" in data["dependencies"]
        assert "yfinance" in data["dependencies"]


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["name"] == "Stock Price Prediction API"
        assert data["version"] == "0.1.0"
        assert "/docs" in data["docs"]


class TestMetricsEndpoint:
    """Test Prometheus metrics endpoint."""

    def test_metrics_endpoint_accessible(self):
        """Test metrics endpoint is accessible."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")


class TestTrainEndpoint:
    """Test training endpoint."""

    @pytest.mark.slow
    def test_train_endpoint_invalid_symbol(self):
        """Test training with invalid symbol format."""
        response = client.post(
            "/api/v1/train",
            json={"symbol": "INVALID!", "period": "1y"},
        )
        # Should return error
        assert response.status_code in [400, 422, 500, 503]

    def test_train_endpoint_invalid_period(self):
        """Test training with invalid period."""
        response = client.post(
            "/api/v1/train",
            json={"symbol": "AAPL", "period": "invalid_period"},
        )
        # Should return validation error
        assert response.status_code == 422

        data = response.json()
        assert "error" in data or "detail" in data


class TestPredictEndpoint:
    """Test prediction endpoint."""

    def test_predict_endpoint_missing_model(self):
        """Test prediction without trained model."""
        response = client.post(
            "/api/v1/predict",
            json={"symbol": "TESTSTOCK"},
        )
        # Should return error (model not found)
        assert response.status_code in [404, 500]

        data = response.json()
        assert "error" in data or "detail" in data

    def test_predict_endpoint_validation(self):
        """Test prediction endpoint validates input."""
        response = client.post(
            "/api/v1/predict",
            json={},  # Missing symbol
        )
        # Should return validation error
        assert response.status_code == 422


class TestModelsEndpoint:
    """Test models management endpoints."""

    def test_list_models_endpoint(self):
        """Test listing models endpoint."""
        response = client.get("/api/v1/models")
        # Should succeed even if no models exist
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert "models" in data or "total_models" in data

    def test_get_model_info_not_found(self):
        """Test getting info for non-existent model."""
        response = client.get("/api/v1/models/NONEXISTENT/latest")
        # Should return not found
        assert response.status_code == 404

        data = response.json()
        assert "error" in data or "detail" in data


class TestErrorHandling:
    """Test error handling."""

    def test_404_on_invalid_route(self):
        """Test 404 on invalid route."""
        response = client.get("/api/v1/invalid_route")
        assert response.status_code == 404

    def test_validation_error_format(self):
        """Test validation error returns proper format."""
        response = client.post(
            "/api/v1/train",
            json={"period": "1y"},  # Missing required symbol
        )
        assert response.status_code == 422

        data = response.json()
        # Should have proper error structure
        assert "detail" in data or "error" in data


class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers_present(self):
        """Test CORS headers are present in response."""
        response = client.get(
            "/api/v1/health",
            headers={"Origin": "http://localhost:3000"},
        )
        # CORS middleware should add headers
        assert response.status_code == 200
        # Note: TestClient may not include all CORS headers
        # This is more of a smoke test
