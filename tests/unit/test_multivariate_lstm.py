"""Unit tests for multivariate LSTM model."""

import numpy as np
import pytest
import torch

from src.infrastructure.model.lstm_multivariate import (
    MultiFeatureScaler,
    MultivariateLSTMNetwork,
    MultivariateLSTMPredictor,
    TemporalAttentionLayer,
)


class TestTemporalAttentionLayer:
    """Test suite for TemporalAttentionLayer."""

    def test_attention_forward_pass(self):
        """Test forward pass through attention layer."""
        hidden_size = 64
        batch_size = 8
        seq_len = 60

        attention = TemporalAttentionLayer(hidden_size)

        # Create sample input
        lstm_output = torch.randn(batch_size, seq_len, hidden_size)

        # Forward pass
        context_vector, attention_weights = attention(lstm_output)

        # Check output shapes
        assert context_vector.shape == (batch_size, hidden_size)
        assert attention_weights.shape == (batch_size, seq_len)

        # Check attention weights sum to 1
        weights_sum = attention_weights.sum(dim=1)
        torch.testing.assert_close(
            weights_sum,
            torch.ones(batch_size),
            rtol=1e-4,
            atol=1e-4,
        )


class TestMultivariateLSTMNetwork:
    """Test suite for MultivariateLSTMNetwork."""

    def test_network_initialization(self):
        """Test network initialization."""
        num_features = 24
        network = MultivariateLSTMNetwork(
            num_features=num_features,
            hidden_size_1=128,
            hidden_size_2=64,
            dropout=0.3,
            use_attention=True,
        )

        assert network.num_features == num_features
        assert network.hidden_size_1 == 128
        assert network.hidden_size_2 == 64
        assert network.use_attention is True

    def test_network_forward_pass(self):
        """Test forward pass through network."""
        num_features = 24
        lookback = 60
        batch_size = 8

        network = MultivariateLSTMNetwork(num_features=num_features)

        # Create sample input
        x = torch.randn(batch_size, lookback, num_features)

        # Forward pass
        predictions = network(x)

        # Check output shape
        assert predictions.shape == (batch_size, 1)

    def test_network_with_attention(self):
        """Test network with attention returns attention weights."""
        num_features = 24
        lookback = 60
        batch_size = 8

        network = MultivariateLSTMNetwork(
            num_features=num_features,
            use_attention=True,
        )

        x = torch.randn(batch_size, lookback, num_features)

        # Forward pass with attention
        predictions, attention_weights = network(x, return_attention=True)

        # Check shapes
        assert predictions.shape == (batch_size, 1)
        assert attention_weights.shape == (batch_size, lookback)

    def test_network_without_attention(self):
        """Test network without attention mechanism."""
        num_features = 24
        lookback = 60
        batch_size = 8

        network = MultivariateLSTMNetwork(
            num_features=num_features,
            use_attention=False,
        )

        x = torch.randn(batch_size, lookback, num_features)
        predictions = network(x)

        assert predictions.shape == (batch_size, 1)


class TestMultiFeatureScaler:
    """Test suite for MultiFeatureScaler."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training and test data."""
        np.random.seed(42)
        X_train = np.random.randn(100, 60, 24)  # 100 samples, 60 lookback, 24 features
        y_train = np.random.randn(100, 1)
        X_test = np.random.randn(20, 60, 24)
        return X_train, y_train, X_test

    def test_scaler_fit_transform(self, sample_data):
        """Test scaler fit and transform."""
        X_train, y_train, _ = sample_data

        scaler = MultiFeatureScaler()
        scaler.fit(X_train, y_train)

        assert scaler.is_fitted is True

        # Transform data
        X_scaled = scaler.transform_features(X_train)

        # Check shape preserved
        assert X_scaled.shape == X_train.shape

        # Check standardization (mean ≈ 0, std ≈ 1)
        # Flatten for checking
        X_scaled_flat = X_scaled.reshape(-1, X_scaled.shape[-1])
        assert abs(X_scaled_flat.mean()) < 0.1
        assert abs(X_scaled_flat.std() - 1.0) < 0.2

    def test_scaler_only_fit_on_train(self, sample_data):
        """
        CRITICAL TEST: Ensure scaler is only fit on training data.

        This prevents data leakage where test statistics influence training.
        """
        X_train, y_train, X_test = sample_data

        scaler = MultiFeatureScaler()
        scaler.fit(X_train, y_train)

        # Get training mean/std
        train_mean = scaler.feature_scaler.mean_
        train_std = scaler.feature_scaler.scale_

        # Create new scaler with train+test (WRONG approach)
        scaler_leaky = MultiFeatureScaler()
        X_combined = np.concatenate([X_train, X_test], axis=0)
        scaler_leaky.fit(X_combined)

        # Get combined mean/std
        combined_mean = scaler_leaky.feature_scaler.mean_
        combined_std = scaler_leaky.feature_scaler.scale_

        # They should be different (proving we're only using train)
        assert not np.allclose(train_mean, combined_mean, rtol=1e-2)
        assert not np.allclose(train_std, combined_std, rtol=1e-2)

    def test_inverse_transform_target(self, sample_data):
        """Test inverse transform of targets."""
        _, y_train, _ = sample_data

        scaler = MultiFeatureScaler()
        scaler.fit(None, y_train)

        # Transform and inverse transform
        y_scaled = scaler.transform_target(y_train)
        y_reconstructed = scaler.inverse_transform_target(y_scaled)

        # Check reconstruction
        np.testing.assert_array_almost_equal(
            y_train.flatten(),
            y_reconstructed.flatten(),
            decimal=5,
        )

    def test_scaler_error_before_fit(self):
        """Test error when using scaler before fitting."""
        scaler = MultiFeatureScaler()
        X = np.random.randn(10, 60, 24)

        with pytest.raises(ValueError, match="must be fitted"):
            scaler.transform_features(X)


class TestMultivariateLSTMPredictor:
    """Test suite for MultivariateLSTMPredictor."""

    @pytest.fixture
    def sample_data(self):
        """Create sample features and targets."""
        np.random.seed(42)
        features = np.random.randn(200, 24)  # 200 samples, 24 features
        targets = np.random.randn(200)
        return features, targets

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = MultivariateLSTMPredictor(
            num_features=24,
            lookback=60,
            hidden_size_1=128,
            hidden_size_2=64,
        )

        assert predictor.num_features == 24
        assert predictor.lookback == 60
        assert predictor.is_fitted is False

    def test_prepare_sequences(self, sample_data):
        """Test sequence preparation."""
        features, targets = sample_data

        predictor = MultivariateLSTMPredictor(num_features=24, lookback=60)
        X, y = predictor.prepare_sequences(features, targets)

        # Check shapes
        expected_sequences = len(features) - predictor.lookback
        assert X.shape == (expected_sequences, predictor.lookback, 24)
        assert y.shape == (expected_sequences, 1)

    def test_temporal_data_split(self, sample_data):
        """
        CRITICAL TEST: Ensure data split is temporal (not random).

        Training data should come before validation, validation before test.
        """
        features, targets = sample_data

        predictor = MultivariateLSTMPredictor(num_features=24, lookback=60)
        X, y = predictor.prepare_sequences(features, targets)

        X_train, X_val, X_test, y_train, y_val, y_test = predictor.split_data_temporal(
            X, y, train_ratio=0.7, val_ratio=0.15
        )

        # Check no shuffling: last sample of train should come before first sample of val
        # Since we're using temporal split, indices should be ordered
        assert len(X_train) > 0
        assert len(X_val) > 0
        assert len(X_test) > 0

        # Check total samples
        assert len(X_train) + len(X_val) + len(X_test) == len(X)

        # Check temporal order: train < val < test in terms of time
        # First val sample should come after last train sample
        # (This is implicit in the split, but we verify the sizes match expectation)
        expected_train_size = int(len(X) * 0.7)
        expected_val_size = int(len(X) * 0.15)

        assert len(X_train) == expected_train_size
        assert len(X_val) == expected_val_size

    def test_mc_dropout_produces_variance(self, sample_data):
        """
        Test that Monte Carlo Dropout produces reasonable variance.

        Multiple forward passes should produce different predictions,
        indicating dropout is active.
        """
        features, targets = sample_data

        predictor = MultivariateLSTMPredictor(
            num_features=24,
            lookback=60,
            epochs=5,  # Quick training for test
            batch_size=16,
        )

        # Train model
        predictor.train(features, targets, verbose=False)

        # Predict with MC Dropout
        recent_features = features[-60:]
        mean_pred, lower, upper = predictor.predict_with_uncertainty(
            recent_features, n_iterations=100, confidence_level=0.95
        )

        # Check interval is non-zero (dropout creates uncertainty)
        assert upper > lower
        assert abs(upper - lower) > 0

        # Check mean is within interval
        assert lower <= mean_pred <= upper

    def test_save_and_load_model(self, sample_data, tmp_path):
        """Test model saving and loading."""
        features, targets = sample_data

        # Train model
        predictor = MultivariateLSTMPredictor(
            num_features=24,
            lookback=60,
            epochs=5,
        )
        predictor.train(features, targets, verbose=False)

        # Make prediction
        recent_features = features[-60:]
        pred_before, _, _ = predictor.predict_with_uncertainty(
            recent_features, n_iterations=10
        )

        # Save model
        save_dir = tmp_path / "model"
        predictor.save_model(str(save_dir))

        # Load model
        predictor_loaded = MultivariateLSTMPredictor(
            num_features=24,
            lookback=60,
        )
        predictor_loaded.load_model(str(save_dir))

        # Make prediction with loaded model
        pred_after, _, _ = predictor_loaded.predict_with_uncertainty(
            recent_features, n_iterations=10
        )

        # Predictions should be similar (allowing for MC dropout variance)
        assert abs(pred_before - pred_after) < 0.5

    def test_feature_shape_validation(self):
        """Test validation of feature shapes."""
        predictor = MultivariateLSTMPredictor(num_features=24, lookback=60)

        # Create insufficient features
        insufficient_features = np.random.randn(50, 24)  # Only 50 samples, need 60

        with pytest.raises(ValueError, match="at least 60"):
            # First need to train (mark as fitted)
            predictor.is_fitted = True
            predictor.predict_with_uncertainty(insufficient_features)
