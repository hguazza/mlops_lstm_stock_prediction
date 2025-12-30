# Multivariate LSTM Model Guide

Comprehensive guide for the multivariate LSTM stock return prediction model.

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Feature Engineering](#feature-engineering)
4. [Training Process](#training-process)
5. [Prediction & Uncertainty](#prediction--uncertainty)
6. [Data Leakage Prevention](#data-leakage-prevention)
7. [API Usage](#api-usage)
8. [Performance Metrics](#performance-metrics)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The multivariate LSTM model predicts **stock returns (%)** for a target ticker using historical data from 4 input tickers as features. Unlike traditional univariate models that only use the target stock's history, this approach captures market relationships and cross-asset dynamics.

### Key Features

- **Multivariate Input**: Uses 4 related stocks as predictive features
- **Technical Indicators**: Calculates RSI, MACD, volatility, and volume indicators
- **Temporal Attention**: Learns which historical periods are most relevant
- **Uncertainty Quantification**: Monte Carlo Dropout provides confidence intervals
- **No Data Leakage**: Strict temporal validation ensures production reliability

### Output

The model predicts **returns in percentage**:
- Positive value (e.g., `+2.5%`): Expected price increase
- Negative value (e.g., `-1.2%`): Expected price decrease
- Confidence interval: Range of likely outcomes (e.g., `[0.5%, 4.2%]` at 95% confidence)

---

## Model Architecture

### Network Structure

```
Input: (batch_size, lookback=60, num_features=24)
    ↓
LSTM Layer 1 (hidden_size=128, return_sequences=True)
    ↓
Temporal Attention Layer
    ↓
LSTM Layer 2 (hidden_size=64, return_sequences=False)
    ↓
Dropout (0.3) ← Used in MC Dropout for uncertainty
    ↓
Dense Layer (32 units) + ReLU
    ↓
Dropout (0.2)
    ↓
Dense Layer (1 unit) → Predicted Return (%)
```

### Components

#### 1. LSTM Layers

- **Layer 1**: Processes entire sequence, captures temporal patterns
- **Layer 2**: Refines representation after attention

#### 2. Temporal Attention

Learns to focus on relevant time steps:
```python
attention_weights = softmax(W * tanh(V * lstm_output))
context = sum(attention_weights * lstm_output)
```

**Benefits**:
- Focuses on important market events
- Improves interpretability (can visualize which days mattered)
- Boosts performance on long sequences

#### 3. Dropout Layers

- **Training**: Randomly drops neurons (regularization)
- **MC Dropout**: Keeps dropout active during inference to quantify uncertainty

---

## Feature Engineering

### Input Features (24 Total)

For each of the 4 input tickers, we calculate:

1. **Log Returns** (`ticker_return`)
   - Formula: `log(price_t / price_t-1)`
   - Why: Stationary, symmetric, additive over time

2. **RSI (14 periods)** (`ticker_rsi`)
   - Formula: `RSI = 100 - 100/(1 + RS)` where `RS = Avg Gain / Avg Loss`
   - Range: 0-100 (normalized to [-1, 1] for neural network)
   - Interpretation: > 70 overbought, < 30 oversold

3. **MACD Histogram** (`ticker_macd`)
   - Formula: `MACD_line - Signal_line`
   - Indicates momentum and trend changes
   - Normalized by price std

4. **Realized Volatility** (`ticker_volatility`)
   - Formula: Rolling std of returns (20-day window)
   - Measures price uncertainty/risk

5. **Normalized Volume** (`ticker_volume`)
   - Formula: `log(volume / rolling_mean(volume, 20))`
   - Identifies unusual trading activity

### Why These Features?

| Feature | Purpose | Market Signal |
|---------|---------|---------------|
| Log Returns | Price movements | Trend direction |
| RSI | Momentum | Overbought/oversold conditions |
| MACD | Trend changes | Buy/sell signals |
| Volatility | Risk measure | Market uncertainty |
| Volume | Activity level | Conviction/strength |

### Target Variable

- **Target**: Log return of target ticker at time `t+forecast_horizon`
- **Shifted**: Features at time `t` use only data up to `t-1` (prevents leakage)

---

## Training Process

### 1. Data Preparation

```python
# Fetch data for all tickers
tickers = input_tickers + [target_ticker]
data = fetch_multiple_tickers(tickers, periods=365)  # ~1 year

# Align dates (intersection)
aligned_data = align_dates(data)

# Calculate features
features, targets = calculate_features(aligned_data)
```

### 2. Temporal Split

**CRITICAL**: Data is split temporally, NOT randomly.

```python
# Temporal split (70% train, 15% val, 15% test)
train_end = int(len(data) * 0.70)
val_end = int(len(data) * 0.85)

train_data = data[:train_end]
val_data = data[train_end:val_end]
test_data = data[val_end:]
```

**Why?** Time series have temporal dependence. Random splits leak future information.

### 3. Normalization

```python
# Fit scaler ONLY on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Transform all data using train statistics
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)  # Uses train mean/std
```

**Why?** Test/production data shouldn't influence training statistics.

### 4. Training Loop

```python
for epoch in range(max_epochs):
    # Training
    for batch in train_loader:
        predictions = model(batch_X)
        loss = MSE(predictions, batch_y)
        loss.backward()
        optimizer.step()

    # Validation
    val_loss = validate(model, val_loader)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

**Hyperparameters**:
- Learning rate: 0.001 (Adam optimizer)
- Batch size: 32
- Max epochs: 100
- Early stopping patience: 15 epochs
- Gradient clipping: 1.0

---

## Prediction & Uncertainty

### Monte Carlo Dropout

Traditional neural networks give point estimates. MC Dropout provides uncertainty:

```python
def predict_with_uncertainty(features, n_iterations=100):
    model.train()  # Enable dropout
    predictions = []

    for _ in range(n_iterations):
        pred = model(features)
        predictions.append(pred)

    mean = np.mean(predictions)
    lower = np.percentile(predictions, 2.5)   # 95% CI
    upper = np.percentile(predictions, 97.5)

    return mean, lower, upper
```

**Interpretation**:
- **Mean**: Best estimate of return
- **Interval Width**: Model confidence (wide = uncertain, narrow = confident)
- **Asymmetry**: If interval is asymmetric, model sees skewed risk

**Example**:
```
Predicted Return: 2.34%
Confidence Interval: [0.5%, 4.2%] at 95% confidence

Interpretation:
- Expected return is +2.34%
- 95% chance actual return is between 0.5% and 4.2%
- Model is fairly confident (interval width = 3.7%)
```

---

## Data Leakage Prevention

### Critical Rules

1. **Feature Shift**: Features at time `t` use only data up to `t-1`
   ```python
   features_shifted = features.shift(1)  # Prevent leakage
   ```

2. **Temporal Split**: Training data comes before validation/test
   ```python
   # CORRECT
   train = data[:70%]
   val = data[70%:85%]
   test = data[85%:]

   # WRONG (random split)
   train, val, test = random_split(data)  # ❌ Leakage!
   ```

3. **Scaler Fitting**: Fit only on training data
   ```python
   # CORRECT
   scaler.fit(X_train)
   X_val_scaled = scaler.transform(X_val)

   # WRONG
   scaler.fit(X_train + X_val)  # ❌ Leakage!
   ```

4. **No Future Targets**: Target at time `t` predicts `t+horizon`, not `t`

### Testing for Leakage

Run unit tests:
```bash
pytest tests/unit/test_technical_indicators.py::test_no_future_data_in_features
pytest tests/unit/test_multivariate_lstm.py::test_scaler_only_fit_on_train
pytest tests/unit/test_multivariate_lstm.py::test_temporal_data_split
```

---

## API Usage

### Endpoint: POST /api/v1/multivariate/train-predict

Train model and generate prediction in one call.

**Request**:
```json
{
  "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
  "target_ticker": "NVDA",
  "lookback": 60,
  "forecast_horizon": 5,
  "confidence_level": 0.95,
  "period": "1y",
  "config": {
    "hidden_size": 128,
    "epochs": 100,
    "batch_size": 32
  }
}
```

**Response**:
```json
{
  "status": "success",
  "model_id": "multivariate_predictor_NVDA",
  "mlflow_run_id": "abc123def456",
  "training_metrics": {
    "best_val_loss": 0.0156,
    "mae": 0.0234,
    "rmse": 0.0312,
    "directional_accuracy": 0.64,
    "epochs_trained": 45,
    "training_time_seconds": 120.5
  },
  "prediction": {
    "target_ticker": "NVDA",
    "input_tickers": ["META", "GOOG", "TSLA", "BABA"],
    "predicted_return_pct": 2.34,
    "confidence_interval": {
      "lower": 0.5,
      "upper": 4.2,
      "confidence_level": 0.95
    },
    "forecast_horizon_days": 5,
    "features_used": ["META_return", "META_rsi", ...]
  }
}
```

### cURL Example

```bash
curl -X POST "http://localhost:8000/api/v1/multivariate/train-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "input_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "target_ticker": "META",
    "lookback": 60,
    "forecast_horizon": 5,
    "confidence_level": 0.95,
    "period": "1y"
  }'
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/multivariate/train-predict",
    json={
        "input_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "target_ticker": "META",
        "lookback": 60,
        "forecast_horizon": 5,
        "confidence_level": 0.95,
        "period": "1y",
    }
)

data = response.json()
print(f"Predicted Return: {data['prediction']['predicted_return_pct']:.2f}%")
print(f"Confidence Interval: [{data['prediction']['confidence_interval']['lower']:.2f}%, "
      f"{data['prediction']['confidence_interval']['upper']:.2f}%]")
```

---

## Performance Metrics

### Regression Metrics

- **MAE (Mean Absolute Error)**: Average prediction error in percentage points
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **MSE Loss**: Training objective (minimized during training)

### Directional Accuracy

More relevant for trading: Did we predict the correct direction?

```
Directional Accuracy = % of predictions where sign(predicted) == sign(actual)
```

**Example**:
- Predicted: +2.5%, Actual: +1.2% → ✅ Correct direction
- Predicted: +2.5%, Actual: -0.5% → ❌ Wrong direction

**Interpretation**:
- 50%: Random guessing
- 55-60%: Good performance
- 65%+: Excellent performance

### Training Metrics

Monitor during training:
- **Validation Loss**: Should decrease and stabilize
- **Train/Val Gap**: Large gap indicates overfitting
- **Early Stopping**: Triggers when validation loss stops improving

---

## Troubleshooting

### Issue: Training Loss Not Decreasing

**Possible Causes**:
1. Learning rate too high or too low
2. Insufficient data
3. Features not normalized

**Solutions**:
```python
# Adjust learning rate
config = {"learning_rate": 0.0001}  # Try lower

# Increase data period
request = {"period": "2y"}  # More data

# Check normalization
print(f"Feature mean: {features.mean()}")  # Should be ~0
print(f"Feature std: {features.std()}")    # Should be ~1
```

### Issue: Wide Confidence Intervals

**Possible Causes**:
1. Model is uncertain (high variance in MC Dropout)
2. Insufficient training
3. Noisy data

**Solutions**:
```python
# Train longer
config = {"epochs": 150, "early_stopping_patience": 20}

# Increase model capacity
config = {"hidden_size_1": 256, "hidden_size_2": 128}

# More iterations in MC Dropout
# (In code: n_iterations=200 instead of 100)
```

### Issue: Poor Directional Accuracy (<55%)

**Possible Causes**:
1. Input tickers not correlated with target
2. Market regime change
3. Data quality issues

**Solutions**:
```python
# Choose correlated tickers (same sector)
# Example: For NVDA (semiconductors), use AMD, INTC, TSM, AVGO

# Use longer lookback
request = {"lookback": 120}  # Capture more patterns

# Try different period
request = {"period": "6mo"}  # More recent market conditions
```

### Issue: Predictions Too Close to Zero

**Possible Causes**:
1. Model playing it safe (predicting mean)
2. Insufficient model capacity
3. Overregularization (dropout too high)

**Solutions**:
```python
# Reduce dropout
config = {"dropout": 0.2}  # Lower from 0.3

# Increase model size
config = {"hidden_size_1": 256, "hidden_size_2": 128}

# Check if model is training
# MAE and RMSE should be << 1.0
```

### Issue: MLflow Tracking Not Working

**Check**:
```bash
# Verify MLflow server is running
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Check environment variables
echo $MLFLOW_TRACKING_URI

# View logs
docker-compose logs mlflow
```

### Issue: Out of Memory (OOM)

**Solutions**:
```python
# Reduce batch size
config = {"batch_size": 16}  # Lower from 32

# Reduce lookback
request = {"lookback": 40}  # Lower from 60

# Use CPU instead of GPU
# (Automatically selected if no GPU available)
```

---

## Best Practices

### 1. Ticker Selection

Choose input tickers that are:
- **Correlated** with target ticker (same sector, related businesses)
- **Liquid** (high trading volume)
- **Available** (consistent historical data)

**Example Combinations**:
- Target: NVDA → Inputs: AMD, INTC, TSM, AVGO (semiconductors)
- Target: AAPL → Inputs: MSFT, GOOGL, AMZN, META (big tech)
- Target: JPM → Inputs: BAC, WFC, C, GS (banks)

### 2. Hyperparameter Tuning

Start with defaults, then tune:
1. **Epochs**: Start with 100, increase if underfitting
2. **Hidden Size**: 128/64 is good baseline, increase for complex patterns
3. **Dropout**: 0.3 is safe, reduce if underfit, increase if overfit
4. **Learning Rate**: 0.001 works well, reduce if training unstable

### 3. Production Deployment

Before deploying:
1. **Validate**: Test on out-of-sample data
2. **Monitor**: Track directional accuracy over time
3. **Retrain**: Retrain periodically (e.g., monthly) as markets change
4. **Alert**: Set up alerts for unusual predictions or confidence intervals

### 4. Interpreting Results

- **High Confidence, Small Interval**: Model is confident (good)
- **Low Confidence, Wide Interval**: Model is uncertain (risky)
- **Asymmetric Interval**: Model sees skewed risk distribution
- **Prediction Close to Zero**: No clear signal, market may be neutral

---

## References

### Research Papers

1. **Attention in LSTMs**: "Attention Is All You Need" (Vaswani et al., 2017)
2. **MC Dropout**: "Dropout as a Bayesian Approximation" (Gal & Ghahramani, 2016)
3. **Financial Time Series**: "Deep Learning for Stock Prediction" (Fischer & Krauss, 2018)

### Implementation Details

- PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- Technical Indicators: https://www.investopedia.com/technical-analysis/

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [Create an issue](https://github.com/your-repo/issues)
- Documentation: [Full docs](https://your-docs-site.com)
- API Reference: http://localhost:8000/docs
