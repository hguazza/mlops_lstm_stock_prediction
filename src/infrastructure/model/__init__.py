"""Model infrastructure - ML models and configurations."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json


@dataclass
class ModelConfig:
    """
    Configuration for LSTM stock prediction model.

    Default values match project specification:
    - input_size=1 (univariate: close price only)
    - hidden_size=50 (empirically tested)
    - num_layers=2 (balance complexity/overfitting)
    - sequence_length=60 (60 timesteps as per spec)

    All parameters are configurable for experimentation.
    """

    # Model architecture
    input_size: int = 1
    hidden_size: int = 50
    num_layers: int = 2
    dropout: float = 0.2
    sequence_length: int = 60

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10

    # Data split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Other settings
    gradient_clip: float = 1.0
    device: Optional[str] = None  # None = auto-detect

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config() -> ModelConfig:
    """
    Get default model configuration.

    Returns baseline configuration matching project specification.
    """
    return ModelConfig()


def create_config_variations() -> List[ModelConfig]:
    """
    Create a list of config variations for hyperparameter search.

    Returns:
        List of ModelConfig objects with different hyperparameters
    """
    variations = []

    # Baseline (default)
    variations.append(ModelConfig())

    # Vary hidden size
    for hidden_size in [32, 64, 128]:
        variations.append(ModelConfig(hidden_size=hidden_size))

    # Vary num_layers
    for num_layers in [1, 3]:
        variations.append(ModelConfig(num_layers=num_layers))

    # Vary dropout
    for dropout in [0.1, 0.3]:
        variations.append(ModelConfig(dropout=dropout))

    # Vary learning rate
    for lr in [0.0001, 0.01]:
        variations.append(ModelConfig(learning_rate=lr))

    return variations


__all__ = ["ModelConfig", "get_default_config", "create_config_variations"]
