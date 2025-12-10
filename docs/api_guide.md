# Stock Price Prediction API Guide

Complete guide for using the Stock Price Prediction REST API.

## Table of Contents

1. [Getting Started](#getting-started)
2. [API Endpoints](#api-endpoints)
3. [Request/Response Examples](#requestresponse-examples)
4. [Error Handling](#error-handling)
5. [Monitoring](#monitoring)
6. [Deployment](#deployment)

## Getting Started

### Starting the API

**Development Mode:**
```bash
make run-api
```

**Production Mode:**
```bash
make run-api-prod
```

The API will be available at:
- **Base URL:** `http://localhost:8000`
- **API Documentation:** `http://localhost:8000/docs` (Swagger UI)
- **ReDoc:** `http://localhost:8000/redoc`
- **Metrics:** `http://localhost:8000/metrics` (Prometheus format)

### Environment Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Key environment variables:
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `MLFLOW_TRACKING_URI`: MLflow backend store
- `MODEL_HIDDEN_SIZE`: Default LSTM hidden size
- `MODEL_EPOCHS`: Default training epochs

## API Endpoints

All endpoints are prefixed with `/api/v1`.

### 1. Health Check

**GET** `/api/v1/health`

Check API and dependencies status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-12-10T14:30:52Z",
  "dependencies": {
    "mlflow": "connected",
    "yfinance": "accessible"
  }
}
```

### 2. Train Model

**POST** `/api/v1/train`

Train a new LSTM model for a stock symbol.

**Request:**
```json
{
  "symbol": "AAPL",
  "period": "1y",
  "config": {
    "hidden_size": 50,
    "num_layers": 2,
    "epochs": 100
  }
}
```

**Parameters:**
- `symbol` (required): Stock symbol (e.g., "AAPL", "GOOGL")
- `period` (optional): Historical data period. Valid: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`. Default: `1y`
- `config` (optional): Model configuration overrides
  - `hidden_size`: LSTM hidden size (16-512)
  - `num_layers`: Number of LSTM layers (1-5)
  - `dropout`: Dropout rate (0.0-0.9)
  - `sequence_length`: Input sequence length (10-120)
  - `learning_rate`: Learning rate (0.0-1.0)
  - `batch_size`: Batch size (8-256)
  - `epochs`: Training epochs (1-1000)
  - `early_stopping_patience`: Early stopping patience (1-100)

**Response:**
```json
{
  "status": "success",
  "model_id": "stock_predictor_AAPL:latest",
  "mlflow_run_id": "abc123def456",
  "training_metrics": {
    "best_val_loss": 0.0234,
    "final_train_loss": 0.0189,
    "final_val_loss": 0.0245,
    "epochs_trained": 45,
    "training_time_seconds": 120.5
  },
  "model_info": {
    "model_id": "stock_predictor_AAPL:latest",
    "symbol": "AAPL",
    "trained_at": "2024-12-10T14:30:52Z",
    "data_rows": 252,
    "config": {
      "hidden_size": 50,
      "num_layers": 2,
      "epochs": 100
    },
    "mlflow_run_id": "abc123def456"
  }
}
```

**Status Codes:**
- `201`: Model trained successfully
- `422`: Validation error (invalid parameters)
- `500`: Training failed
- `503`: Data fetch error

### 3. Predict Price

**POST** `/api/v1/predict`

Generate prediction using existing model.

**Request:**
```json
{
  "symbol": "AAPL",
  "model_version": "latest"
}
```

**Parameters:**
- `symbol` (required): Stock symbol
- `model_version` (optional): Model version ("latest", "Production", or version number). Default: "latest"

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "symbol": "AAPL",
    "predicted_price": 178.45,
    "confidence_interval": {
      "lower": 175.20,
      "upper": 181.70,
      "confidence_level": 0.95
    },
    "prediction_date": "2024-12-11",
    "current_price": 176.80
  },
  "model_info": {
    "model_id": "stock_predictor_AAPL:3",
    "symbol": "AAPL",
    "trained_at": "2024-12-10T14:30:52Z",
    "data_rows": 252,
    "config": {"hidden_size": 50, "num_layers": 2},
    "model_stage": "Production"
  }
}
```

**Status Codes:**
- `200`: Prediction successful
- `404`: Model not found
- `422`: Validation error
- `500`: Prediction failed

### 4. Train and Predict

**POST** `/api/v1/predict/train-predict`

Complete pipeline: train new model and generate prediction.

**Request:** Same as `/train` endpoint

**Response:**
```json
{
  "status": "success",
  "model_id": "stock_predictor_AAPL:latest",
  "mlflow_run_id": "xyz789abc123",
  "training_metrics": {
    "best_val_loss": 0.0198,
    "final_train_loss": 0.0175,
    "final_val_loss": 0.0201,
    "epochs_trained": 52,
    "training_time_seconds": 135.2
  },
  "prediction": {
    "symbol": "AAPL",
    "predicted_price": 179.20,
    "confidence_interval": {
      "lower": 176.10,
      "upper": 182.30,
      "confidence_level": 0.95
    },
    "prediction_date": "2024-12-11",
    "current_price": 177.50
  },
  "model_info": {
    "model_id": "stock_predictor_AAPL:latest",
    "symbol": "AAPL",
    "trained_at": "2024-12-10T15:22:10Z",
    "data_rows": 252,
    "config": {"hidden_size": 50, "num_layers": 2}
  }
}
```

### 5. List Models

**GET** `/api/v1/models`

List all registered models from MLflow Registry.

**Response:**
```json
{
  "status": "success",
  "total_models": 2,
  "models": [
    {
      "symbol": "AAPL",
      "versions": 5,
      "latest_version": "5",
      "production_version": "3",
      "last_trained": "2024-12-10T14:30:52Z"
    },
    {
      "symbol": "GOOGL",
      "versions": 2,
      "latest_version": "2",
      "production_version": null,
      "last_trained": "2024-12-09T10:15:20Z"
    }
  ]
}
```

### 6. Get Model Info

**GET** `/api/v1/models/{symbol}/latest`

Get detailed information about the latest model for a symbol.

**Response:**
```json
{
  "status": "success",
  "model_info": {
    "model_id": "stock_predictor_AAPL:5",
    "symbol": "AAPL",
    "trained_at": "2024-12-10T14:30:52Z",
    "data_rows": 252,
    "config": {
      "hidden_size": 50,
      "num_layers": 2,
      "epochs": 100
    },
    "mlflow_run_id": "abc123def456",
    "model_stage": "Staging"
  },
  "metrics": {
    "best_val_loss": 0.0234,
    "final_train_loss": 0.0189,
    "final_val_loss": 0.0245,
    "epochs_trained": 45
  }
}
```

## Request/Response Examples

### Training with Custom Configuration

```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "period": "2y",
    "config": {
      "hidden_size": 64,
      "num_layers": 3,
      "epochs": 50,
      "learning_rate": 0.0005
    }
  }'
```

### Generating Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "model_version": "latest"
  }'
```

### Checking Health

```bash
curl http://localhost:8000/api/v1/health
```

### Listing All Models

```bash
curl http://localhost:8000/api/v1/models
```

## Error Handling

All errors follow a consistent format:

```json
{
  "status": "error",
  "error": "ErrorType",
  "message": "Human-readable error message",
  "details": {
    "key": "additional context"
  },
  "timestamp": "2024-12-10T14:30:52.123456"
}
```

### Error Types

- **ValidationError** (422): Invalid request parameters
- **ModelNotFoundError** (404): Model not found in registry
- **TrainingError** (500): Model training failed
- **PredictionError** (500): Prediction generation failed
- **DataFetchError** (503): Failed to fetch stock data
- **InternalServerError** (500): Unexpected error

### Example Error Response

```json
{
  "status": "error",
  "error": "ValidationError",
  "message": "Invalid stock symbol format",
  "details": {
    "symbol": "INVALID!",
    "expected_format": "uppercase letters"
  },
  "timestamp": "2024-12-10T14:30:52.123456"
}
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics` in Prometheus format.

**Key Metrics:**

1. **HTTP Requests:**
   - `http_requests_total{method, endpoint, status}`: Total requests
   - `http_request_duration_seconds{method, endpoint}`: Request duration

2. **Predictions:**
   - `predictions_total{symbol, status}`: Total predictions
   - `model_load_duration_seconds{symbol}`: Model loading time

3. **Training:**
   - `training_total{symbol, status}`: Total training requests

### Health Checks

Use `/api/v1/health` for:
- Kubernetes liveness/readiness probes
- Load balancer health checks
- Monitoring system checks

## Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install UV
RUN pip install uv

# Copy project files
COPY . .

# Install dependencies
RUN uv sync --no-dev

# Expose port
EXPOSE 8000

# Run API
CMD ["uv", "run", "uvicorn", "src.presentation.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stock-prediction-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stock-prediction-api
  template:
    metadata:
      labels:
        app: stock-prediction-api
    spec:
      containers:
      - name: api
        image: stock-prediction-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "sqlite:///mlflow.db"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

### Production Considerations

1. **Authentication:** Add JWT or API key authentication
2. **Rate Limiting:** Implement rate limiting per client
3. **Caching:** Cache predictions for frequently requested symbols
4. **Scaling:** Use horizontal pod autoscaling based on CPU/memory
5. **Monitoring:** Set up Prometheus + Grafana for metrics visualization
6. **Logging:** Configure structured logging to centralized system (ELK, DataDog)
7. **SSL/TLS:** Enable HTTPS in production
8. **CORS:** Configure specific origins instead of allowing all

## Best Practices

1. **Training:**
   - Train with at least 1 year of historical data
   - Use larger `hidden_size` and `num_layers` for better accuracy
   - Monitor validation loss to avoid overfitting

2. **Prediction:**
   - Retrain models periodically (weekly/monthly)
   - Use Production stage models for critical predictions
   - Consider confidence intervals when making decisions

3. **Model Management:**
   - Tag models with metadata (accuracy, date, etc.)
   - Promote models to Production after validation
   - Keep multiple versions for rollback capability

4. **Performance:**
   - Use `/predict` for existing models (faster)
   - Use `/train-predict` for one-off requests
   - Cache frequently accessed predictions

## Support

For issues or questions:
- Check API documentation at `/docs`
- Review MLflow tracking UI at `http://localhost:5000`
- Check application logs for detailed error information
