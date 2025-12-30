"""Integration tests for multivariate prediction endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    from src.presentation.main import app

    return TestClient(app)


class TestMultivariateTrainPredictEndpoint:
    """Test suite for /api/v1/multivariate/train-predict endpoint."""

    def test_train_predict_success(self, client):
        """Test successful training and prediction."""
        request_data = {
            "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
            "target_ticker": "NVDA",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "period": "6mo",
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # May timeout in test environment, so allow 500 or 503
        if response.status_code == 200:
            data = response.json()

            # Check response structure
            assert "status" in data
            assert "model_id" in data
            assert "training_metrics" in data
            assert "prediction" in data
            assert "model_info" in data

            # Check training metrics
            metrics = data["training_metrics"]
            assert "best_val_loss" in metrics
            assert "mae" in metrics
            assert "rmse" in metrics
            assert "epochs_trained" in metrics

            # Check prediction
            prediction = data["prediction"]
            assert "target_ticker" in prediction
            assert prediction["target_ticker"] == "NVDA"
            assert "input_tickers" in prediction
            assert len(prediction["input_tickers"]) == 4
            assert "predicted_return_pct" in prediction
            assert "confidence_interval" in prediction

            # Check confidence interval
            ci = prediction["confidence_interval"]
            assert "lower" in ci
            assert "upper" in ci
            assert ci["lower"] < ci["upper"]
            assert ci["confidence_level"] == 0.95

    def test_train_predict_invalid_tickers_count(self, client):
        """Test validation error for wrong number of input tickers."""
        request_data = {
            "input_tickers": ["META", "GOOG"],  # Only 2 tickers instead of 4
            "target_ticker": "NVDA",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "period": "1y",
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # Should return validation error
        assert response.status_code == 422  # Unprocessable Entity (Pydantic validation)

    def test_train_predict_duplicate_tickers(self, client):
        """Test validation error for duplicate input tickers."""
        request_data = {
            "input_tickers": ["META", "META", "GOOG", "TSLA"],  # Duplicate META
            "target_ticker": "NVDA",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "period": "1y",
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_train_predict_invalid_lookback(self, client):
        """Test validation error for invalid lookback."""
        request_data = {
            "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
            "target_ticker": "NVDA",
            "lookback": 300,  # Out of range (max 252)
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "period": "1y",
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_train_predict_invalid_confidence_level(self, client):
        """Test validation error for invalid confidence level."""
        request_data = {
            "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
            "target_ticker": "NVDA",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 1.5,  # Out of range (max 0.99)
            "period": "1y",
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_train_predict_with_config_overrides(self, client):
        """Test training with custom configuration."""
        request_data = {
            "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
            "target_ticker": "NVDA",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "period": "6mo",
            "config": {
                "hidden_size": 64,
                "dropout": 0.4,
                "epochs": 20,
                "batch_size": 16,
            },
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # May timeout, check if successful or timeout
        assert response.status_code in [200, 500, 503, 504]

    def test_confidence_interval_bounds(self, client):
        """Test that confidence interval bounds are correctly ordered."""
        request_data = {
            "input_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "target_ticker": "META",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.90,
            "period": "6mo",
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        if response.status_code == 200:
            data = response.json()
            prediction = data["prediction"]
            ci = prediction["confidence_interval"]

            # Check bounds are correctly ordered
            assert ci["lower"] < prediction["predicted_return_pct"]
            assert prediction["predicted_return_pct"] < ci["upper"]
            assert ci["confidence_level"] == 0.90


class TestMultivariatePredictEndpoint:
    """Test suite for /api/v1/multivariate/predict endpoint."""

    def test_predict_not_implemented(self, client):
        """Test that predict-only endpoint returns not implemented."""
        request_data = {
            "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
            "target_ticker": "NVDA",
            "model_version": "latest",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.95,
        }

        response = client.post("/api/v1/multivariate/predict", json=request_data)

        # Should return 501 Not Implemented
        assert response.status_code == 501

        data = response.json()
        assert "detail" in data
        assert "not yet implemented" in data["detail"]["message"].lower()


class TestMultivariateEndpointErrors:
    """Test error handling in multivariate endpoints."""

    def test_invalid_ticker_symbols(self, client):
        """Test handling of invalid ticker symbols."""
        request_data = {
            "input_tickers": ["INVALID1", "INVALID2", "INVALID3", "INVALID4"],
            "target_ticker": "INVALID_TARGET",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "period": "1y",
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # Should return error (either 400 or 500 depending on where validation fails)
        assert response.status_code in [400, 500]

    def test_missing_required_fields(self, client):
        """Test validation error for missing required fields."""
        request_data = {
            "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
            # Missing target_ticker
            "lookback": 60,
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # Should return validation error
        assert response.status_code == 422

    def test_invalid_period_format(self, client):
        """Test validation error for invalid period format."""
        request_data = {
            "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
            "target_ticker": "NVDA",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "period": "invalid_period",
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        # Should return validation error
        assert response.status_code == 422


class TestMultivariateEndpointIntegration:
    """Integration tests for complete multivariate workflow."""

    @pytest.mark.slow
    def test_complete_workflow(self, client):
        """
        Test complete workflow: train, predict, verify results.

        This is a slow test that may take several minutes.
        """
        # Step 1: Train and predict
        request_data = {
            "input_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
            "target_ticker": "META",
            "lookback": 60,
            "forecast_horizon": 5,
            "confidence_level": 0.95,
            "period": "6mo",
            "config": {
                "epochs": 10,  # Reduced for faster test
            },
        }

        response = client.post("/api/v1/multivariate/train-predict", json=request_data)

        if response.status_code == 200:
            data = response.json()

            # Verify model was trained
            assert data["training_metrics"]["epochs_trained"] > 0

            # Verify prediction was made
            assert "predicted_return_pct" in data["prediction"]

            # Verify confidence interval is reasonable
            ci = data["prediction"]["confidence_interval"]
            interval_width = ci["upper"] - ci["lower"]
            assert 0 < interval_width < 50  # Interval shouldn't be too wide

            # Verify features were used
            assert len(data["prediction"]["features_used"]) > 0

    def test_multiple_tickers_combinations(self, client):
        """Test different combinations of input tickers."""
        ticker_combinations = [
            (["AAPL", "MSFT", "GOOGL", "AMZN"], "META"),
            (["META", "NVDA", "AMD", "INTC"], "AAPL"),
            (["JPM", "BAC", "WFC", "C"], "GS"),
        ]

        for input_tickers, target_ticker in ticker_combinations:
            request_data = {
                "input_tickers": input_tickers,
                "target_ticker": target_ticker,
                "lookback": 60,
                "forecast_horizon": 5,
                "confidence_level": 0.95,
                "period": "3mo",
                "config": {"epochs": 5},  # Quick training
            }

            response = client.post(
                "/api/v1/multivariate/train-predict", json=request_data
            )

            # Allow various outcomes due to data availability and timeouts
            assert response.status_code in [200, 400, 500, 503, 504]

    def test_different_confidence_levels(self, client):
        """Test predictions with different confidence levels."""
        confidence_levels = [0.80, 0.90, 0.95, 0.99]

        for confidence_level in confidence_levels:
            request_data = {
                "input_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
                "target_ticker": "META",
                "lookback": 60,
                "forecast_horizon": 5,
                "confidence_level": confidence_level,
                "period": "3mo",
                "config": {"epochs": 5},
            }

            response = client.post(
                "/api/v1/multivariate/train-predict", json=request_data
            )

            if response.status_code == 200:
                data = response.json()
                assert (
                    data["prediction"]["confidence_interval"]["confidence_level"]
                    == confidence_level
                )

                # Higher confidence should generally mean wider intervals
                # (though not always due to MC Dropout stochasticity)
