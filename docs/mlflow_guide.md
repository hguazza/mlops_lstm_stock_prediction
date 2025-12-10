# MLflow Experiment Tracking Guide

This guide explains how to use MLflow for experiment tracking, model registry, and hyperparameter tuning in the Stock Prediction LSTM project.

## Table of Contents

- [Setup](#setup)
- [Basic Usage](#basic-usage)
- [Viewing Experiments](#viewing-experiments)
- [Model Registry](#model-registry)
- [Comparing Experiments](#comparing-experiments)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Best Practices](#best-practices)

## Setup

### Initialize MLflow

Run the initialization script to set up MLflow tracking:

```bash
python scripts/init_mlflow.py
```

This will:
- Create the SQLite tracking database (`mlflow.db`)
- Set up the default experiment (`stock_prediction_lstm`)
- Create artifact storage directory
- Display the MLflow UI command

### Environment Variables

Configure MLflow using environment variables:

```bash
# MLflow settings
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export MLFLOW_EXPERIMENT_NAME="stock_prediction_lstm"
export MLFLOW_ARTIFACT_LOCATION="./mlruns_artifacts"

# Model hyperparameters
export MODEL_HIDDEN_SIZE=50
export MODEL_NUM_LAYERS=2
export MODEL_EPOCHS=100
```

## Basic Usage

### Training with MLflow Tracking

```python
from src.application.services import create_prediction_service_with_mlflow
from src.application.services.data_service import DataService

# Create services
data_service = DataService()
prediction_service = create_prediction_service_with_mlflow()

# Fetch stock data
data = data_service.fetch_stock_data("AAPL", period="1y")

# Train with automatic MLflow tracking
history = prediction_service.train_model(
    data=data,
    target_column="Close",
    verbose=True,
    symbol="AAPL",  # Used for run naming and model registry
)
```

### What Gets Logged Automatically

When you train a model with MLflow enabled, the following are automatically logged:

**Parameters:**
- `input_size`, `hidden_size`, `num_layers`
- `dropout`, `sequence_length`
- `learning_rate`, `batch_size`, `epochs`
- `early_stopping_patience`, `gradient_clip`
- All train/val/test split ratios

**Metrics (per epoch):**
- `train_loss` - Training loss
- `val_loss` - Validation loss

**Final Metrics:**
- `best_val_loss` - Best validation loss achieved
- `final_train_loss` - Final training loss
- `final_val_loss` - Final validation loss
- `epochs_trained` - Number of epochs trained
- `data_rows` - Number of data rows used

**Artifacts:**
- PyTorch model with signature
- Model registered to MLflow Model Registry
- Input/output schema for validation

**Tags:**
- `model_type`: "LSTM"
- `framework`: "PyTorch"
- `symbol`: Stock symbol (e.g., "AAPL")
- `target_column`: Prediction target

## Viewing Experiments

### Launch MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://localhost:5000 in your browser.

### UI Features

- **Experiments Tab**: View all runs in an experiment
- **Compare Runs**: Select multiple runs to compare metrics
- **Run Details**: Click a run to see parameters, metrics, artifacts
- **Metric Plots**: Visualize training curves
- **Model Registry**: Browse registered models and versions

## Model Registry

### Automatic Registration

Models are automatically registered when trained with a symbol:

```python
# Automatically registers as "stock_predictor_AAPL"
history = prediction_service.train_model(data, symbol="AAPL")
```

### Loading from Registry

```python
from src.infrastructure.mlflow.utils import load_model_from_registry

# Load production model
model = load_model_from_registry("stock_predictor_AAPL", stage="Production")

# Load specific version
model = load_model_from_registry("stock_predictor_AAPL", stage="3")
```

### Model Stages

Transition models through stages:

```python
from src.infrastructure.mlflow.mlflow_service import MLflowService
from src.infrastructure.mlflow import MLflowConfig

mlflow_service = MLflowService(MLflowConfig())
mlflow_service.setup_tracking()

# Move model to staging
mlflow_service.transition_model_stage(
    model_name="stock_predictor_AAPL",
    version=2,
    stage="Staging"
)

# Promote to production
mlflow_service.transition_model_stage(
    model_name="stock_predictor_AAPL",
    version=2,
    stage="Production"
)
```

Available stages:
- `Staging` - Testing phase
- `Production` - Live deployment
- `Archived` - Deprecated models

## Comparing Experiments

### Find Best Model

```python
from src.infrastructure.mlflow.utils import get_best_model

# Get model with lowest validation loss
best_model = get_best_model(
    experiment_name="stock_prediction_lstm",
    metric="best_val_loss",
    ascending=True  # Lower is better for loss
)

print(f"Best Run ID: {best_model['run_id']}")
print(f"Best Val Loss: {best_model['metrics']['best_val_loss']}")
print(f"Parameters: {best_model['params']}")
```

### Compare Multiple Runs

```python
from src.infrastructure.mlflow.utils import compare_runs

# Compare specific runs
df = compare_runs(
    run_ids=["run_id_1", "run_id_2", "run_id_3"],
    metrics=["best_val_loss", "final_train_loss"]
)

print(df)
```

### Get All Experiment Metrics

```python
from src.infrastructure.mlflow.utils import get_experiment_metrics

# Get all runs from an experiment
df = get_experiment_metrics(
    experiment_name="stock_prediction_lstm",
    max_results=100
)

# Analyze results
print(df.describe())
print(df.sort_values("best_val_loss").head())
```

## Hyperparameter Tuning

### Grid Search Example

```python
from src.infrastructure.model import ModelConfig, create_config_variations
from src.application.services import create_prediction_service_with_mlflow

# Create service with MLflow
prediction_service = create_prediction_service_with_mlflow()

# Define parameter grid
param_grid = {
    "hidden_size": [32, 50, 64],
    "num_layers": [1, 2, 3],
    "dropout": [0.1, 0.2, 0.3],
}

# Generate configurations
configs = create_config_variations(param_grid)

# Train all configurations
results = []
for i, config in enumerate(configs):
    print(f"\nTraining configuration {i+1}/{len(configs)}")

    # Update model config
    prediction_service.config = config
    prediction_service.model = LSTMStockPredictor(config=config)

    # Train with MLflow tracking
    history = prediction_service.train_model(
        data=data,
        symbol="AAPL",
        verbose=False,
    )

    results.append({
        "config": config.to_dict(),
        "best_val_loss": min(history["val_loss"]),
    })

# Find best configuration
best_result = min(results, key=lambda x: x["best_val_loss"])
print(f"\nBest Configuration: {best_result['config']}")
print(f"Best Val Loss: {best_result['best_val_loss']}")
```

### Random Search

```python
import random

# Randomly sample hyperparameters
for run in range(20):
    config = ModelConfig(
        hidden_size=random.choice([32, 50, 64, 128]),
        num_layers=random.choice([1, 2, 3]),
        dropout=random.uniform(0.1, 0.5),
        learning_rate=random.uniform(0.0001, 0.01),
    )

    prediction_service.config = config
    prediction_service.model = LSTMStockPredictor(config=config)

    history = prediction_service.train_model(data, symbol="AAPL")
```

## Best Practices

### Experiment Naming

Use descriptive experiment names:
- `stock_prediction_lstm` - Main experiment
- `stock_prediction_lstm_hyperparam_tuning` - Hyperparameter search
- `stock_prediction_lstm_daily_retrain` - Production retraining

### Run Naming

Include key information in run names:
- `lstm_AAPL_20241210_143052` - Auto-generated with timestamp
- `experiment_hidden128_layers3` - Manual naming for experiments

### Tagging Strategy

Add relevant tags to runs:
```python
tags = {
    "model_type": "LSTM",
    "symbol": "AAPL",
    "data_period": "1y",
    "purpose": "hyperparameter_tuning",
    "engineer": "data_science_team",
}

prediction_service.mlflow_service.set_tags(tags)
```

### Model Registry Workflow

1. **Development**: Train multiple models, track in MLflow
2. **Staging**: Promote best model to `Staging` stage
3. **Testing**: Validate model in staging environment
4. **Production**: Transition to `Production` stage
5. **Monitoring**: Track production model performance
6. **Archival**: Move old models to `Archived`

### Cleanup

Delete old experiment runs:
```python
from src.infrastructure.mlflow.utils import delete_experiment_runs

# Requires explicit confirmation
deleted_count = delete_experiment_runs(
    experiment_name="test_experiment",
    confirm=True
)
```

## Troubleshooting

### Database Locked Error

If you see "database is locked":
- Close MLflow UI
- Ensure no other processes are accessing the database
- Wait a few seconds and retry

### Missing Artifacts

If artifacts are not found:
- Check `artifact_location` path in config
- Ensure directory has write permissions
- Verify artifacts were logged successfully

### PyTorch Model Loading

If model loading fails:
- Ensure same PyTorch version
- Check model signature compatibility
- Verify all dependencies are installed

## Advanced Usage

### Custom Callbacks

Create custom epoch callbacks for specialized logging:

```python
def custom_callback(event, epoch, train_loss, val_loss, **kwargs):
    if event == "epoch" and epoch % 5 == 0:
        # Log additional metrics
        mlflow_service.log_metrics({
            "loss_ratio": train_loss / val_loss,
            "epoch": epoch,
        }, step=epoch)

history = predictor.train(
    data=data,
    epoch_callback=custom_callback
)
```

### Parallel Experimentation

Run multiple experiments in parallel using separate experiments or run IDs.

### Integration with CI/CD

Include MLflow tracking in automated training pipelines:
- Log metrics to compare against baseline
- Automatically promote models that meet criteria
- Trigger alerts for performance degradation

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
- [Model Registry](https://mlflow.org/docs/latest/model-registry.html)
