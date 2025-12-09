"""LSTM model implementation for stock price prediction."""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from . import ModelConfig

logger = structlog.get_logger(__name__)


class LSTMNetwork(nn.Module):
    """
    PyTorch LSTM neural network for time series prediction.

    Architecture:
        Input: (batch_size, sequence_length, input_size)
        → LSTM layers (with dropout)
        → Fully connected layer
        → Output: (batch_size, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        """
        Initialize LSTM network.

        Args:
            input_size: Number of input features (1 for univariate)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
        """
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor (batch_size, sequence_length, input_size)

        Returns:
            Output predictions (batch_size, 1)
        """
        # LSTM forward pass
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take only the last output
        last_output = lstm_out[:, -1, :]

        # Fully connected layer
        predictions = self.fc(last_output)

        return predictions


class LSTMStockPredictor:
    """
    LSTM-based stock price predictor with complete training and inference pipeline.

    Implements:
    - Configurable LSTM architecture (default: input=1, hidden=50, layers=2)
    - Data preprocessing (normalization, sequence generation)
    - Training with early stopping
    - Prediction with confidence
    - Model persistence
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize LSTM stock predictor.

        Args:
            config: Model configuration (uses defaults if None)
        """
        self.config = config or ModelConfig()
        self.logger = logger.bind(component="LSTMStockPredictor")

        # Initialize model
        self.model = LSTMNetwork(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        # Set device
        if self.config.device:
            self.device = torch.device(self.config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Scaler for normalization
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Training state
        self.is_fitted = False
        self.training_history = {"train_loss": [], "val_loss": []}

        self.logger.info(
            "lstm_predictor_initialized",
            config=self.config.to_dict(),
            device=str(self.device),
        )

    def prepare_sequences(
        self, data: np.ndarray, sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences from time series data using sliding window.

        Args:
            data: Time series data (n_samples,)
            sequence_length: Number of timesteps per sequence

        Returns:
            X: Input sequences (n_sequences, sequence_length, 1)
            y: Target values (n_sequences, 1)
        """
        X, y = [], []

        for i in range(len(data) - sequence_length):
            sequence = data[i : i + sequence_length]
            target = data[i + sequence_length]
            X.append(sequence)
            y.append(target)

        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y).reshape(-1, 1)

        return X, y

    def split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train, validation, and test sets.

        Args:
            X: Input sequences
            y: Target values

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        n_samples = len(X)
        train_size = int(n_samples * self.config.train_ratio)
        val_size = int(n_samples * self.config.val_ratio)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size : train_size + val_size]
        y_val = y[train_size : train_size + val_size]

        X_test = X[train_size + val_size :]
        y_test = y[train_size + val_size :]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training and validation.

        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences
            y_val: Validation targets

        Returns:
            train_loader, val_loader
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False
        )

        return train_loader, val_loader

    def train(
        self,
        data: pd.DataFrame,
        target_column: str = "Close",
        verbose: bool = True,
    ) -> dict:
        """
        Train the LSTM model on stock data.

        Args:
            data: DataFrame with stock prices
            target_column: Column to predict (default: 'Close')
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        self.logger.info("starting_training", data_shape=data.shape)

        # Extract and normalize data
        prices = data[target_column].values.reshape(-1, 1)
        normalized_prices = self.scaler.fit_transform(prices).flatten()

        # Prepare sequences
        X, y = self.prepare_sequences(normalized_prices, self.config.sequence_length)
        self.logger.info("sequences_created", n_sequences=len(X))

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        self.logger.info(
            "data_split",
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test),
        )

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(
            X_train, y_train, X_val, y_val
        )

        # Setup training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            val_loss = self.validate(val_loader, criterion)

            # Save history
            self.training_history["train_loss"].append(avg_train_loss)
            self.training_history["val_loss"].append(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if verbose:
                    self.logger.info(
                        "new_best_model",
                        epoch=epoch + 1,
                        train_loss=avg_train_loss,
                        val_loss=val_loss,
                    )
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                self.logger.info(
                    "training_progress",
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                )

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(
                    "early_stopping",
                    epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                )
                break

        self.is_fitted = True
        self.logger.info("training_completed", best_val_loss=best_val_loss)

        return self.training_history

    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        """
        Validate the model on validation data.

        Args:
            val_loader: Validation DataLoader
            criterion: Loss function

        Returns:
            Average validation loss
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                val_losses.append(loss.item())

        return np.mean(val_losses)

    def predict(self, historical_data: np.ndarray) -> float:
        """
        Predict next price from historical data.

        Args:
            historical_data: Recent prices (sequence_length values)

        Returns:
            Predicted next price
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")

        if len(historical_data) < self.config.sequence_length:
            raise ValueError(
                f"Need at least {self.config.sequence_length} historical data points"
            )

        # Take last sequence_length values
        recent_data = historical_data[-self.config.sequence_length :]

        # Normalize
        normalized = self.scaler.transform(recent_data.reshape(-1, 1)).flatten()

        # Prepare input
        X = normalized.reshape(1, self.config.sequence_length, 1)
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(X_tensor)

        # Denormalize
        prediction_numpy = prediction.cpu().numpy()
        predicted_price = self.scaler.inverse_transform(prediction_numpy)[0][0]

        return float(predicted_price)

    def predict_from_dataframe(
        self, data: pd.DataFrame, target_column: str = "Close"
    ) -> float:
        """
        Predict next price from DataFrame.

        Args:
            data: DataFrame with historical prices
            target_column: Column to use for prediction

        Returns:
            Predicted next price
        """
        prices = data[target_column].values
        return self.predict(prices)

    def save_model(self, save_dir: str) -> None:
        """
        Save complete model state.

        Args:
            save_dir: Directory to save model files
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = save_path / "model.pth"
        torch.save(self.model.state_dict(), model_path)

        # Save scaler
        scaler_path = save_path / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        # Save config
        config_path = save_path / "config.json"
        self.config.to_json(str(config_path))

        # Save training history
        history_path = save_path / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        self.logger.info("model_saved", path=str(save_path))

    def load_model(self, load_dir: str) -> None:
        """
        Load complete model state.

        Args:
            load_dir: Directory containing model files
        """
        load_path = Path(load_dir)

        # Load config
        config_path = load_path / "config.json"
        self.config = ModelConfig.from_json(str(config_path))

        # Recreate model with loaded config
        self.model = LSTMNetwork(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )

        # Load weights
        model_path = load_path / "model.pth"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

        # Load scaler
        scaler_path = load_path / "scaler.pkl"
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # Load history if exists
        history_path = load_path / "history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                self.training_history = json.load(f)

        self.is_fitted = True
        self.logger.info("model_loaded", path=str(load_path))
