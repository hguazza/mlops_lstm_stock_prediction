"""Multivariate LSTM model for stock return prediction.

This module implements a sophisticated LSTM architecture for predicting stock returns
using multiple ticker features with:
- Temporal attention mechanism
- Monte Carlo Dropout for uncertainty quantification
- Proper normalization with StandardScaler
- Prevention of data leakage
"""

import json
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

logger = structlog.get_logger(__name__)


class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention layer for LSTM.

    Learns to focus on relevant time steps in the sequence,
    improving model performance on time series with varying importance.

    Architecture:
        Input: (batch_size, seq_len, hidden_size)
        → Linear projection to attention scores
        → Softmax to get attention weights
        → Weighted sum of inputs
        → Output: (batch_size, hidden_size)
    """

    def __init__(self, hidden_size: int):
        """
        Initialize attention layer.

        Args:
            hidden_size: Size of LSTM hidden state
        """
        super(TemporalAttentionLayer, self).__init__()

        self.hidden_size = hidden_size

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention to LSTM output.

        Args:
            lstm_output: LSTM output (batch_size, seq_len, hidden_size)

        Returns:
            Tuple of:
            - context_vector: Weighted sum of inputs (batch_size, hidden_size)
            - attention_weights: Attention weights (batch_size, seq_len)
        """
        # Calculate attention scores: (batch, seq_len, 1)
        attention_scores = self.attention(lstm_output)

        # Softmax over sequence dimension: (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Weighted sum: (batch, hidden_size)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        # Squeeze for output: (batch, seq_len)
        attention_weights = attention_weights.squeeze(-1)

        return context_vector, attention_weights


class MultivariateLSTMNetwork(nn.Module):
    """
    Multivariate LSTM network with attention for return prediction.

    Architecture:
        Input: (batch_size, lookback, num_features)
        → LSTM Layer 1 (hidden_size=128, return_sequences=True)
        → Temporal Attention
        → LSTM Layer 2 (hidden_size=64, return_sequences=False)
        → Dropout (0.3) - used in MC Dropout
        → Dense (32) + ReLU
        → Dropout (0.2)
        → Dense (1) - output: predicted return (%)
    """

    def __init__(
        self,
        num_features: int,
        hidden_size_1: int = 128,
        hidden_size_2: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        """
        Initialize multivariate LSTM network.

        Args:
            num_features: Number of input features (e.g., 24 for 4 tickers × 6 features)
            hidden_size_1: Size of first LSTM layer (default: 128)
            hidden_size_2: Size of second LSTM layer (default: 64)
            dropout: Dropout rate for MC Dropout (default: 0.3)
            use_attention: Whether to use attention mechanism (default: True)
        """
        super(MultivariateLSTMNetwork, self).__init__()

        self.num_features = num_features
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.dropout_rate = dropout
        self.use_attention = use_attention

        # First LSTM layer (returns sequences for attention)
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0,  # No dropout in LSTM itself
        )

        # Temporal attention layer
        if use_attention:
            self.attention = TemporalAttentionLayer(hidden_size_1)

        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )

        # Fully connected layers
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size_2, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.67)  # Slightly lower
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, lookback, num_features)
            return_attention: If True, return attention weights (for visualization)

        Returns:
            If return_attention=False:
                Predicted returns (batch_size, 1)
            If return_attention=True:
                Tuple of (predictions, attention_weights)
        """
        # First LSTM layer: (batch, seq_len, hidden_size_1)
        lstm1_out, _ = self.lstm1(x)

        # Apply attention if enabled
        attention_weights = None
        if self.use_attention:
            # Attention: (batch, hidden_size_1)
            context, attention_weights = self.attention(lstm1_out)
            # Expand for LSTM2: (batch, 1, hidden_size_1)
            lstm2_input = context.unsqueeze(1)
        else:
            # Without attention, use last output
            lstm2_input = lstm1_out

        # Second LSTM layer: (batch, seq_len, hidden_size_2)
        lstm2_out, _ = self.lstm2(lstm2_input)

        # Take last output: (batch, hidden_size_2)
        last_output = lstm2_out[:, -1, :]

        # Fully connected layers with dropout
        x = self.dropout1(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        predictions = self.fc2(x)

        if return_attention and attention_weights is not None:
            return predictions, attention_weights

        return predictions


class MultiFeatureScaler:
    """
    StandardScaler for multivariate features with separate target scaling.

    Critical: This scaler MUST be fit only on training data to prevent leakage.
    """

    def __init__(self):
        """Initialize scaler."""
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> None:
        """
        Fit scaler on training data only.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples, 1), optional
        """
        # Reshape if needed
        if X_train.ndim == 3:
            # Shape: (n_samples, lookback, n_features) → (n_samples * lookback, n_features)
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            self.feature_scaler.fit(X_train_reshaped)
        else:
            self.feature_scaler.fit(X_train)

        if y_train is not None:
            self.target_scaler.fit(y_train.reshape(-1, 1))

        self.is_fitted = True

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Args:
            X: Features to transform (n_samples, lookback, n_features)
               or (n_samples, n_features)

        Returns:
            Scaled features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")

        if X.ndim == 3:
            original_shape = X.shape
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.feature_scaler.transform(X_reshaped)
            return X_scaled.reshape(original_shape)
        else:
            return self.feature_scaler.transform(X)

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """Transform target using fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        return self.target_scaler.transform(y.reshape(-1, 1))

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform target back to original scale."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse transform")
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1))


class MultivariateLSTMPredictor:
    """
    Complete predictor for multivariate LSTM with training and inference.

    Implements:
    - Multivariate feature processing
    - LSTM training with validation
    - Monte Carlo Dropout for uncertainty quantification
    - Model persistence
    """

    def __init__(
        self,
        num_features: int,
        lookback: int = 60,
        hidden_size_1: int = 128,
        hidden_size_2: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        device: Optional[str] = None,
    ):
        """
        Initialize predictor.

        Args:
            num_features: Number of input features
            lookback: Sequence length (default: 60 days)
            hidden_size_1: First LSTM hidden size (default: 128)
            hidden_size_2: Second LSTM hidden size (default: 64)
            dropout: Dropout rate (default: 0.3)
            use_attention: Use attention mechanism (default: True)
            learning_rate: Learning rate (default: 0.001)
            batch_size: Batch size (default: 32)
            epochs: Maximum epochs (default: 100)
            early_stopping_patience: Early stopping patience (default: 15)
            device: Device to use ('cpu', 'cuda', or None for auto)
        """
        self.num_features = num_features
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        self.logger = logger.bind(component="MultivariateLSTMPredictor")

        # Initialize model
        self.model = MultivariateLSTMNetwork(
            num_features=num_features,
            hidden_size_1=hidden_size_1,
            hidden_size_2=hidden_size_2,
            dropout=dropout,
            use_attention=use_attention,
        )

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Scaler
        self.scaler = MultiFeatureScaler()

        # Training state
        self.is_fitted = False
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
        }

        self.logger.info(
            "multivariate_lstm_initialized",
            num_features=num_features,
            lookback=lookback,
            device=str(self.device),
            use_attention=use_attention,
        )

    def prepare_sequences(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from feature matrix and targets.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)

        Returns:
            Tuple of:
            - X: Input sequences (n_sequences, lookback, n_features)
            - y: Target values (n_sequences, 1)
        """
        X, y = [], []

        for i in range(len(features) - self.lookback):
            sequence = features[i : i + self.lookback]
            target = targets[i + self.lookback]
            X.append(sequence)
            y.append(target)

        X = np.array(X)
        y = np.array(y).reshape(-1, 1)

        return X, y

    def split_data_temporal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data temporally (no shuffling).

        CRITICAL: This is a temporal split, NOT random split.
        Training data comes before validation, validation before test.

        Args:
            X: Input sequences
            y: Target values
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n_samples = len(X)
        train_size = int(n_samples * train_ratio)
        val_size = int(n_samples * val_ratio)

        # Temporal split
        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size : train_size + val_size]
        y_val = y[train_size : train_size + val_size]

        X_test = X[train_size + val_size :]
        y_test = y[train_size + val_size :]

        self.logger.info(
            "data_split_temporal",
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test),
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        verbose: bool = True,
        epoch_callback: Optional[Callable] = None,
    ) -> Dict:
        """
        Train the multivariate LSTM model.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target values (n_samples,)
            verbose: Print training progress
            epoch_callback: Optional callback after each epoch

        Returns:
            Training history dictionary
        """
        self.logger.info(
            "starting_training",
            features_shape=features.shape,
            targets_shape=targets.shape,
        )

        # Prepare sequences
        X, y = self.prepare_sequences(features, targets)
        self.logger.info("sequences_created", n_sequences=len(X))

        # Temporal split
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_temporal(X, y)

        # Fit scaler on training data ONLY
        self.scaler.fit(X_train, y_train)
        self.logger.info("scaler_fitted_on_train_data")

        # Transform all data
        X_train_scaled = self.scaler.transform_features(X_train)
        X_val_scaled = self.scaler.transform_features(X_val)
        y_train_scaled = self.scaler.transform_target(y_train)
        y_val_scaled = self.scaler.transform_target(y_val)

        # Create DataLoaders
        train_loader = self._create_dataloader(
            X_train_scaled, y_train_scaled, shuffle=True
        )
        val_loader = self._create_dataloader(X_val_scaled, y_val_scaled, shuffle=False)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training phase
            train_loss, train_mae = self._train_epoch(
                train_loader, criterion, optimizer
            )

            # Validation phase
            val_loss, val_mae = self._validate_epoch(val_loader, criterion)

            # Save history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["train_mae"].append(train_mae)
            self.training_history["val_mae"].append(val_mae)

            # Callback
            if epoch_callback:
                try:
                    epoch_callback(
                        event="epoch",
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        train_mae=train_mae,
                        val_mae=val_mae,
                    )
                except Exception as e:
                    self.logger.warning(
                        "callback_failed", epoch=epoch + 1, error=str(e)
                    )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if verbose and (epoch + 1) % 10 == 0:
                    self.logger.info(
                        "new_best_model",
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_mae=val_mae,
                    )
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_patience:
                self.logger.info(
                    "early_stopping",
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                )
                break

        self.is_fitted = True
        self.logger.info("training_completed", best_val_loss=best_val_loss)

        return self.training_history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        train_losses = []
        train_maes = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            predictions = self.model(batch_X)
            loss = criterion(predictions, batch_y)

            # Calculate MAE
            mae = torch.mean(torch.abs(predictions - batch_y))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_maes.append(mae.item())

        return np.mean(train_losses), np.mean(train_maes)

    def _validate_epoch(
        self, val_loader: DataLoader, criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_losses = []
        val_maes = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                mae = torch.mean(torch.abs(predictions - batch_y))

                val_losses.append(loss.item())
                val_maes.append(mae.item())

        return np.mean(val_losses), np.mean(val_maes)

    def _create_dataloader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = False
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def predict_with_uncertainty(
        self,
        features: np.ndarray,
        n_iterations: int = 100,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Predict with uncertainty using Monte Carlo Dropout.

        Args:
            features: Recent feature matrix (lookback, n_features)
            n_iterations: Number of MC iterations (default: 100)
            confidence_level: Confidence level for interval (default: 0.95)

        Returns:
            Tuple of (mean_prediction, lower_bound, upper_bound) in % returns
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        # Prepare input
        if features.shape[0] < self.lookback:
            raise ValueError(
                f"Need at least {self.lookback} samples, got {features.shape[0]}"
            )

        # Take last lookback samples
        recent_features = features[-self.lookback :]

        # Scale features
        X = recent_features.reshape(1, self.lookback, self.num_features)
        X_scaled = self.scaler.transform_features(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Monte Carlo Dropout
        self.model.train()  # Enable dropout
        predictions = []

        with torch.no_grad():
            for _ in range(n_iterations):
                pred = self.model(X_tensor)
                pred_cpu = pred.cpu().numpy()
                predictions.append(pred_cpu[0, 0])

        predictions = np.array(predictions)

        # Inverse transform predictions
        predictions_original = self.scaler.inverse_transform_target(
            predictions.reshape(-1, 1)
        ).flatten()

        # Calculate statistics
        mean_pred = np.mean(predictions_original)
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        lower_bound = np.percentile(predictions_original, lower_percentile)
        upper_bound = np.percentile(predictions_original, upper_percentile)

        self.logger.info(
            "prediction_with_uncertainty",
            mean=mean_pred,
            lower=lower_bound,
            upper=upper_bound,
            std=np.std(predictions_original),
        )

        return float(mean_pred), float(lower_bound), float(upper_bound)

    def save_model(self, save_dir: str) -> None:
        """Save complete model state."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.model.state_dict(), save_path / "model.pth")

        # Save scaler
        with open(save_path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save config
        config = {
            "num_features": self.num_features,
            "lookback": self.lookback,
            "hidden_size_1": self.model.hidden_size_1,
            "hidden_size_2": self.model.hidden_size_2,
            "dropout": self.model.dropout_rate,
            "use_attention": self.model.use_attention,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save training history
        with open(save_path / "history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info("model_saved", path=str(save_path))

    def load_model(self, load_dir: str) -> None:
        """Load complete model state."""
        load_path = Path(load_dir)

        # Load config
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)

        # Recreate model
        self.model = MultivariateLSTMNetwork(
            num_features=config["num_features"],
            hidden_size_1=config["hidden_size_1"],
            hidden_size_2=config["hidden_size_2"],
            dropout=config["dropout"],
            use_attention=config["use_attention"],
        )

        # Load weights
        self.model.load_state_dict(
            torch.load(load_path / "model.pth", map_location=self.device)
        )
        self.model.to(self.device)

        # Load scaler
        with open(load_path / "scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        # Load history
        if (load_path / "history.json").exists():
            with open(load_path / "history.json", "r") as f:
                self.training_history = json.load(f)

        self.is_fitted = True
        self.logger.info("model_loaded", path=str(load_path))
