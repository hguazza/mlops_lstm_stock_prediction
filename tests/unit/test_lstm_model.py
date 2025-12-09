"""Unit tests for LSTM model components."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.infrastructure.model import ModelConfig
from src.infrastructure.model.lstm_model import LSTMNetwork, LSTMStockPredictor


@pytest.mark.unit
class TestLSTMNetwork:
    """Test LSTMNetwork PyTorch module."""

    def test_network_init(self):
        """Test network initialization with parameters."""
        # Arrange & Act
        network = LSTMNetwork(input_size=1, hidden_size=32, num_layers=2, dropout=0.2)

        # Assert
        assert network.hidden_size == 32
        assert network.num_layers == 2
        assert isinstance(network.lstm, torch.nn.LSTM)
        assert isinstance(network.fc, torch.nn.Linear)

    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        # Arrange
        network = LSTMNetwork(input_size=1, hidden_size=16, num_layers=1)
        batch_size, seq_len, input_size = 4, 20, 1
        x = torch.randn(batch_size, seq_len, input_size)

        # Act
        output = network(x)

        # Assert
        assert output.shape == (batch_size, 1)

    def test_hidden_size_configuration(self):
        """Test different hidden sizes."""
        for hidden_size in [16, 32, 64]:
            # Arrange & Act
            network = LSTMNetwork(input_size=1, hidden_size=hidden_size, num_layers=1)

            # Assert
            assert network.hidden_size == hidden_size
            assert network.lstm.hidden_size == hidden_size

    def test_num_layers_configuration(self):
        """Test different number of layers."""
        for num_layers in [1, 2, 3]:
            # Arrange & Act
            network = LSTMNetwork(input_size=1, hidden_size=16, num_layers=num_layers)

            # Assert
            assert network.num_layers == num_layers
            assert network.lstm.num_layers == num_layers


@pytest.mark.unit
class TestLSTMStockPredictorInit:
    """Test LSTMStockPredictor initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        # Arrange & Act
        predictor = LSTMStockPredictor()

        # Assert
        assert predictor.config is not None
        assert predictor.model is not None
        assert predictor.scaler is not None
        assert predictor.is_fitted is False

    def test_init_custom_config(self, lstm_config):
        """Test initialization with custom config."""
        # Arrange & Act
        predictor = LSTMStockPredictor(config=lstm_config)

        # Assert
        assert predictor.config == lstm_config
        assert predictor.model.hidden_size == lstm_config.hidden_size
        assert predictor.model.num_layers == lstm_config.num_layers

    def test_device_selection_cpu(self):
        """Test that device is set correctly (CPU in tests)."""
        # Arrange & Act
        predictor = LSTMStockPredictor()

        # Assert
        assert predictor.device.type in ["cpu", "cuda"]


@pytest.mark.unit
class TestPrepareSequences:
    """Test sequence preparation."""

    def test_prepare_sequences_shape(self):
        """Test that sequences have correct shape."""
        # Arrange
        predictor = LSTMStockPredictor()
        data = np.linspace(100, 200, 100)
        sequence_length = 20

        # Act
        X, y = predictor.prepare_sequences(data, sequence_length)

        # Assert
        expected_samples = len(data) - sequence_length
        assert X.shape == (expected_samples, sequence_length, 1)
        assert y.shape == (expected_samples, 1)

    def test_prepare_sequences_content(self):
        """Test that sequences are created correctly."""
        # Arrange
        predictor = LSTMStockPredictor()
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        sequence_length = 2

        # Act
        X, y = predictor.prepare_sequences(data, sequence_length)

        # Assert
        assert len(X) == 3  # 5 - 2 = 3 sequences
        np.testing.assert_array_equal(X[0].flatten(), [1, 2])
        assert y[0][0] == 3
        np.testing.assert_array_equal(X[1].flatten(), [2, 3])
        assert y[1][0] == 4


@pytest.mark.unit
class TestSplitData:
    """Test data splitting."""

    def test_split_data_ratios(self):
        """Test that data is split according to ratios."""
        # Arrange
        config = ModelConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        predictor = LSTMStockPredictor(config=config)

        X = np.random.rand(100, 20, 1)
        y = np.random.rand(100, 1)

        # Act
        X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data(X, y)

        # Assert
        assert len(X_train) == 70
        assert len(X_val) == 15
        assert len(X_test) == 15
        assert len(y_train) == 70
        assert len(y_val) == 15
        assert len(y_test) == 15


@pytest.mark.unit
class TestCreateDataloaders:
    """Test DataLoader creation."""

    def test_create_dataloaders(self, lstm_config):
        """Test that DataLoaders are created correctly."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)
        X_train = np.random.rand(40, 20, 1)
        y_train = np.random.rand(40, 1)
        X_val = np.random.rand(10, 20, 1)
        y_val = np.random.rand(10, 1)

        # Act
        train_loader, val_loader = predictor.create_dataloaders(
            X_train, y_train, X_val, y_val
        )

        # Assert
        assert train_loader is not None
        assert val_loader is not None
        assert train_loader.batch_size == lstm_config.batch_size


@pytest.mark.unit
@pytest.mark.slow
class TestTrain:
    """Test model training."""

    def test_train_updates_is_fitted(self, small_stock_data, lstm_config):
        """Test that training sets is_fitted to True."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)
        assert predictor.is_fitted is False

        # Act
        predictor.train(small_stock_data, verbose=False)

        # Assert
        assert predictor.is_fitted is True

    def test_train_returns_history(self, small_stock_data, lstm_config):
        """Test that training returns history dict."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)

        # Act
        history = predictor.train(small_stock_data, verbose=False)

        # Assert
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

    def test_train_loss_decreases(self, small_stock_data, lstm_config):
        """Test that training loss generally decreases."""
        # Arrange
        config = ModelConfig(
            hidden_size=16, num_layers=1, epochs=10, early_stopping_patience=10
        )
        predictor = LSTMStockPredictor(config=config)

        # Act
        history = predictor.train(small_stock_data, verbose=False)

        # Assert - first loss should be higher than best loss
        first_loss = history["train_loss"][0]
        best_loss = min(history["train_loss"])
        assert first_loss > best_loss


@pytest.mark.unit
class TestPredict:
    """Test prediction methods."""

    def test_predict_untrained_model_raises_error(self):
        """Test that prediction fails on untrained model."""
        # Arrange
        predictor = LSTMStockPredictor()
        data = np.random.rand(70)

        # Act & Assert
        with pytest.raises(ValueError, match="Model must be trained"):
            predictor.predict(data)

    def test_predict_insufficient_data_raises_error(
        self, small_stock_data, lstm_config
    ):
        """Test that prediction fails with insufficient data."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)
        predictor.train(small_stock_data, verbose=False)

        # Too short data (less than sequence_length)
        short_data = np.random.rand(lstm_config.sequence_length - 1)

        # Act & Assert
        with pytest.raises(ValueError, match="Need at least"):
            predictor.predict(short_data)

    def test_predict_returns_float(self, small_stock_data, lstm_config):
        """Test that prediction returns a float."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)
        predictor.train(small_stock_data, verbose=False)

        prices = small_stock_data["Close"].values

        # Act
        prediction = predictor.predict(prices)

        # Assert
        assert isinstance(prediction, float)
        assert prediction > 0  # Price should be positive

    def test_predict_from_dataframe(self, small_stock_data, lstm_config):
        """Test prediction from DataFrame."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)
        predictor.train(small_stock_data, verbose=False)

        # Act
        prediction = predictor.predict_from_dataframe(small_stock_data)

        # Assert
        assert isinstance(prediction, float)
        assert prediction > 0


@pytest.mark.unit
class TestModelPersistence:
    """Test model save and load."""

    def test_save_model(self, small_stock_data, lstm_config):
        """Test that model can be saved."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)
        predictor.train(small_stock_data, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"

            # Act
            predictor.save_model(str(save_path))

            # Assert
            assert save_path.exists()
            assert (save_path / "model.pth").exists()
            assert (save_path / "scaler.pkl").exists()
            assert (save_path / "config.json").exists()
            assert (save_path / "history.json").exists()

    def test_load_model(self, small_stock_data, lstm_config):
        """Test that model can be loaded."""
        # Arrange
        predictor1 = LSTMStockPredictor(config=lstm_config)
        predictor1.train(small_stock_data, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"

            # Save model
            predictor1.save_model(str(save_path))

            # Create new predictor and load
            predictor2 = LSTMStockPredictor()

            # Act
            predictor2.load_model(str(save_path))

            # Assert
            assert predictor2.is_fitted is True
            assert predictor2.config.hidden_size == lstm_config.hidden_size

    def test_save_load_predictions_match(self, small_stock_data, lstm_config):
        """Test that predictions match after save/load."""
        # Arrange
        predictor1 = LSTMStockPredictor(config=lstm_config)
        predictor1.train(small_stock_data, verbose=False)

        prices = small_stock_data["Close"].values
        pred1 = predictor1.predict(prices)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"

            # Save and load
            predictor1.save_model(str(save_path))
            predictor2 = LSTMStockPredictor()
            predictor2.load_model(str(save_path))

            # Act
            pred2 = predictor2.predict(prices)

            # Assert - predictions should be very close
            assert abs(pred1 - pred2) < 0.01  # Allow small numerical differences


@pytest.mark.unit
class TestScalerState:
    """Test MinMaxScaler state preservation."""

    def test_scaler_fits_during_training(self, small_stock_data, lstm_config):
        """Test that scaler is fitted during training."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)

        # Act
        predictor.train(small_stock_data, verbose=False)

        # Assert
        assert hasattr(predictor.scaler, "data_min_")
        assert hasattr(predictor.scaler, "data_max_")
        assert predictor.scaler.data_min_ is not None

    def test_scaler_used_in_prediction(self, small_stock_data, lstm_config):
        """Test that scaler is used for normalization in prediction."""
        # Arrange
        predictor = LSTMStockPredictor(config=lstm_config)
        predictor.train(small_stock_data, verbose=False)

        prices = small_stock_data["Close"].values

        # Act
        prediction = predictor.predict(prices)

        # Assert - prediction should be in reasonable range
        # (not normalized 0-1 range, should be actual price range)
        min_price = prices.min()
        max_price = prices.max()
        assert min_price * 0.5 < prediction < max_price * 2.0
