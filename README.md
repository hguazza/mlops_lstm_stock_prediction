# Stock Price Prediction API

<div align="center">

![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.0+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-ready-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

LSTM-based stock price prediction REST API with MLflow tracking and Docker deployment.

[Quick Start](#quick-start) •
[Features](#features) •
[API Docs](#api-documentation) •
[Docker](#docker-deployment) •
[Development](#development)

</div>

---

## Features

✨ **Production-Ready ML API**
- 🎯 LSTM model for time series prediction
- 📊 Real-time stock data fetching with `yfinance`
- 🔄 MLflow experiment tracking and model registry
- 📈 Prometheus metrics for monitoring
- 🐳 Docker containerization
- 🧪 Comprehensive test coverage

🏗️ **Clean Architecture**
- Domain-driven design
- Dependency injection
- SOLID principles
- Testable components

🚀 **Developer Experience**
- FastAPI with auto-generated docs
- Structured logging with `structlog`
- Pre-commit hooks for code quality
- Type hints throughout
- Hot reload for development

## Quick Start

### With Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/your-repo/mlops-lstm-stock-prediction.git
cd mlops-lstm-stock-prediction

# Start services
docker-compose up -d

# Access services
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# MLflow UI: http://localhost:5000
```

### Without Docker

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make install

# Initialize MLflow
uv run python scripts/init_mlflow.py

# Run API
make run-api
```

## API Documentation

### Endpoints

#### `POST /api/v1/train`
Train a new LSTM model for a stock symbol.

```bash
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "period": "1y",
    "config": {
      "hidden_size": 50,
      "num_layers": 2,
      "epochs": 100
    }
  }'
```

#### `POST /api/v1/predict`
Generate prediction using existing model.

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "model_version": "latest"
  }'
```

#### `POST /api/v1/predict/train-predict`
Complete pipeline: train and predict in one call.

#### `GET /api/v1/health`
Health check endpoint.

```bash
curl http://localhost:8000/api/v1/health
```

#### `GET /api/v1/models`
List all trained models from MLflow Registry.

#### `GET /api/v1/models/{symbol}/latest`
Get information about the latest model for a symbol.

#### `GET /metrics`
Prometheus metrics endpoint.

### Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Docker Deployment

### Quick Commands

```bash
# Build image
make docker-build

# Start services
make docker-run

# View logs
make docker-logs

# Stop services
make docker-stop

# Check status
make docker-ps

# Clean up
make docker-clean
```

### Docker Compose

**Production:**
```bash
docker-compose up -d
```

**Development (with hot reload):**
```bash
docker-compose -f docker-compose.dev.yml up -d
```

### Architecture

```
┌─────────────────────────────────────────────┐
│         Docker Compose Stack                │
├─────────────────────────────────────────────┤
│                                             │
│  ┌─────────────┐     ┌─────────────┐       │
│  │ API         │     │ MLflow UI   │       │
│  │ Port 8000   │◄────┤ Port 5000   │       │
│  └──────┬──────┘     └──────┬──────┘       │
│         │                   │               │
│         └────────┬──────────┘               │
│                  ▼                          │
│         ┌────────────────┐                  │
│         │ Shared Volume  │                  │
│         │ - mlflow.db    │                  │
│         │ - artifacts    │                  │
│         └────────────────┘                  │
└─────────────────────────────────────────────┘
```

## Development

### Setup

```bash
# Install dependencies
make install

# Install pre-commit hooks
make pre-commit-install

# Run pre-commit checks
make pre-commit-run
```

### Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
make test-coverage

# Run fast tests (excluding slow ones)
make test-fast
```

### Code Quality

```bash
# Format code
make format

# Run linter
make check

# Format + lint
make lint
```

### Project Structure

```
mlops_lstm_stock_prediction/
├── src/
│   ├── domain/              # Domain entities and models
│   ├── application/         # Use cases and services
│   ├── infrastructure/      # External integrations
│   │   ├── data/           # Data loaders
│   │   ├── model/          # LSTM model
│   │   └── mlflow/         # MLflow integration
│   └── presentation/        # FastAPI endpoints
│       ├── api/
│       │   ├── routers/    # API routes
│       │   ├── dependencies.py
│       │   ├── errors.py
│       │   └── middleware/
│       └── schemas/         # Pydantic schemas
├── tests/
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── scripts/                # Helper scripts
├── docs/                   # Documentation
├── Dockerfile              # Multi-stage Dockerfile
├── docker-compose.yml      # Production compose
├── docker-compose.dev.yml  # Development compose
├── Makefile               # Dev commands
└── pyproject.toml         # Project config
```

## Technology Stack

- **Language:** Python 3.12+
- **Package Manager:** UV
- **Web Framework:** FastAPI
- **ML Framework:** PyTorch
- **Data Processing:** Pandas, NumPy
- **Data Validation:** Pydantic, Pandera
- **ML Tracking:** MLflow
- **Monitoring:** Prometheus
- **Logging:** Structlog
- **Testing:** Pytest
- **Code Quality:** Ruff, Pre-commit
- **Containerization:** Docker, Docker Compose

## Configuration

Configuration is managed via environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment variables:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# MLflow Settings
MLFLOW_TRACKING_URI=sqlite:///data/mlflow.db
MLFLOW_EXPERIMENT_NAME=stock_prediction_lstm

# Model Settings
MODEL_HIDDEN_SIZE=50
MODEL_NUM_LAYERS=2
MODEL_EPOCHS=100
```

## MLflow Integration

### View Experiments

```bash
# MLflow UI is available at http://localhost:5000

# Or run standalone
uv run mlflow ui
```

### Model Registry

All trained models are automatically registered in MLflow:

```python
# Load model from registry
from src.infrastructure.mlflow.utils import load_model_from_registry

model = load_model_from_registry("stock_predictor_AAPL", "Production")
```

See [MLflow Guide](docs/mlflow_guide.md) for more details.

## Deployment

See [Deployment Guide](docs/deployment_guide.md) for comprehensive deployment instructions including:

- Local deployment
- Docker deployment
- Kubernetes deployment
- Cloud deployments (AWS, GCP, Azure)
- Monitoring and scaling
- Security best practices

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics`:

- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `predictions_total` - Total predictions
- `training_total` - Total training requests
- `model_load_duration_seconds` - Model loading time

### Health Checks

```bash
curl http://localhost:8000/api/v1/health
```

Response:
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

## Performance

- **Prediction latency:** < 100ms (cached model)
- **Training time:** ~2-5 minutes (1 year data, 100 epochs)
- **Image size:** ~500MB (multi-stage build)
- **Memory usage:** ~512MB - 2GB (depending on model size)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

```bash
# Before committing
make lint
make test
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

- **Documentation:** [docs/](docs/)
- **API Reference:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
- **Issues:** GitHub Issues

## Roadmap

- [ ] Add more model architectures (GRU, Transformer)
- [ ] Implement model versioning and A/B testing
- [ ] Add authentication and rate limiting
- [ ] Integrate with more data sources
- [ ] Add batch prediction endpoint
- [ ] Implement model explanation (SHAP, LIME)
- [ ] Add WebSocket for real-time predictions
- [ ] Create Grafana dashboards

---

**Built with ❤️ using Clean Architecture and MLOps best practices**
