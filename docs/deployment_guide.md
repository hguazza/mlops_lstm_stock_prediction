# Deployment Guide

Complete guide for deploying the Stock Price Prediction API in different environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Scaling](#scaling)

## Prerequisites

### Required Software

- **Docker** >= 20.10
- **Docker Compose** >= 2.0
- **Python** >= 3.12 (for local development)
- **UV** (for local development)

### Installing Docker

**Linux:**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

**macOS:**
```bash
brew install --cask docker
```

**Windows:**
Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)

## Local Development

### Without Docker

```bash
# Install dependencies
make install

# Run MLflow init
uv run python scripts/init_mlflow.py

# Run API
make run-api
```

### With Docker (Development Mode)

```bash
# Build and start development containers
make docker-dev

# View logs
make docker-logs

# Stop services
make docker-stop
```

## Docker Deployment

### Quick Start

```bash
# Build image
make docker-build

# Run services
make docker-run

# Check status
make docker-ps

# View logs
make docker-logs
```

### Manual Docker Build

```bash
# Build the image
docker build -t stock-prediction-api:latest .

# Run the container
docker run -d \
  --name stock-prediction-api \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=sqlite:///data/mlflow.db \
  -v mlflow-data:/data \
  stock-prediction-api:latest
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

### Accessing Services

After starting with docker-compose:

- **API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs
- **MLflow UI:** http://localhost:5000
-   *Dev compose binds MLflow UI to 0.0.0.0 with allowed hosts/CORS open; if you override the command, keep `--host 0.0.0.0 --allowed-hosts 0.0.0.0,127.0.0.1,localhost --cors-allowed-origins '*'` so the UI is reachable from the host.*
- **Prometheus Metrics:** http://localhost:8000/metrics
- **Health Check:** http://localhost:8000/api/v1/health

## Production Deployment

### Environment Variables

Create a `.env` file (based on `.env.example`):

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false

# MLflow Settings
MLFLOW_TRACKING_URI=sqlite:///data/mlflow.db
MLFLOW_EXPERIMENT_NAME=stock_prediction_lstm
MLFLOW_ARTIFACT_LOCATION=/data/mlruns_artifacts

# Model Settings
MODEL_HIDDEN_SIZE=50
MODEL_NUM_LAYERS=2
MODEL_EPOCHS=100
```

### Production Best Practices

#### 1. Use Environment Variables

```bash
docker run -d \
  --name stock-prediction-api \
  -p 8000:8000 \
  --env-file .env \
  -v mlflow-data:/data \
  stock-prediction-api:latest
```

#### 2. Configure Health Checks

Health checks are pre-configured in the Dockerfile:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
```

#### 3. Use Persistent Volumes

```yaml
volumes:
  mlflow-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /path/to/persistent/storage
```

#### 4. Configure Restart Policies

```yaml
services:
  api:
    restart: unless-stopped
```

#### 5. Enable Logging

```bash
# View logs
docker logs -f stock-prediction-api

# Docker compose
docker-compose logs -f api
```

### Kubernetes Deployment

**deployment.yaml:**
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
          value: "sqlite:///data/mlflow.db"
        volumeMounts:
        - name: mlflow-data
          mountPath: /data
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
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
      volumes:
      - name: mlflow-data
        persistentVolumeClaim:
          claimName: mlflow-pvc
```

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: stock-prediction-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: stock-prediction-api
```

### Cloud Deployments

#### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag stock-prediction-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/stock-prediction-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/stock-prediction-api:latest

# Create ECS task definition and service
aws ecs create-service ...
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/<project-id>/stock-prediction-api
gcloud run deploy stock-prediction-api \
  --image gcr.io/<project-id>/stock-prediction-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
# Build and push to ACR
az acr build --registry <registry-name> --image stock-prediction-api:latest .

# Deploy to ACI
az container create \
  --resource-group <resource-group> \
  --name stock-prediction-api \
  --image <registry-name>.azurecr.io/stock-prediction-api:latest \
  --dns-name-label stock-prediction-api \
  --ports 8000
```

## Monitoring

### Prometheus Metrics

Metrics are exposed at `/metrics`:

```bash
curl http://localhost:8000/metrics
```

**Key Metrics:**
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration
- `predictions_total` - Total predictions
- `training_total` - Total training requests
- `model_load_duration_seconds` - Model loading time

### Grafana Dashboard

**docker-compose with Grafana:**
```yaml
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
```

### Logging

**Structured Logs with Structlog:**
```bash
# View logs with docker-compose
docker-compose logs -f api

# Filter by log level
docker-compose logs -f api | grep ERROR
```

### Health Monitoring

**Automated Health Checks:**
```bash
# Check health endpoint
curl http://localhost:8000/api/v1/health

# Response
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

## Troubleshooting

### Common Issues

#### 1. Container Fails to Start

**Check logs:**
```bash
docker logs stock-prediction-api
```

**Verify health:**
```bash
docker inspect --format='{{json .State.Health}}' stock-prediction-api
```

#### 2. Port Already in Use

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
docker run -p 8001:8000 stock-prediction-api:latest
```

#### 3. Volume Permission Issues

```bash
# Fix permissions
sudo chown -R 1000:1000 /path/to/mlflow-data

# Or run as root (not recommended)
docker run --user root ...
```

#### 4. Out of Memory

**Increase Docker memory:**
```bash
# Docker Desktop: Settings > Resources > Memory
# Or limit container memory
docker run --memory="2g" stock-prediction-api:latest
```

#### 5. MLflow Database Locked

```bash
# Stop all containers
docker-compose down

# Remove lock file
rm mlflow.db-journal

# Restart
docker-compose up -d
```

### Debug Mode

**Enable debug logging:**
```bash
docker run -e LOG_LEVEL=DEBUG stock-prediction-api:latest
```

**Access container shell:**
```bash
docker exec -it stock-prediction-api bash
```

## Scaling

### Horizontal Scaling

**Docker Compose:**
```bash
docker-compose up -d --scale api=3
```

**Kubernetes:**
```bash
kubectl scale deployment stock-prediction-api --replicas=5
```

### Load Balancing

**Nginx configuration:**
```nginx
upstream api_backend {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://api_backend;
    }
}
```

### Caching

**Redis for prediction caching:**
```yaml
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### Performance Optimization

1. **Use production ASGI server (already configured):**
   - Uvicorn with 4 workers

2. **Enable caching:**
   - Cache predictions for frequently requested symbols
   - Use Redis or in-memory cache

3. **Database optimization:**
   - Use PostgreSQL instead of SQLite for MLflow
   - Connection pooling

4. **Model optimization:**
   - Quantize models for faster inference
   - Batch predictions when possible

## Security

### Best Practices

1. **Non-root user:** Already configured in Dockerfile
2. **Environment variables:** Never commit `.env` files
3. **HTTPS:** Use reverse proxy with SSL/TLS
4. **Authentication:** Add JWT or API key authentication
5. **Rate limiting:** Implement rate limiting per client
6. **Network isolation:** Use Docker networks

### Adding Authentication

**Example with API Keys:**
```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key
```

## Backup and Recovery

### Backup MLflow Data

```bash
# Backup volume
docker run --rm -v mlflow-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/mlflow-backup.tar.gz /data

# Restore
docker run --rm -v mlflow-data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/mlflow-backup.tar.gz -C /
```

### Database Backup

```bash
# Backup SQLite database
docker exec stock-prediction-api cp /data/mlflow.db /data/mlflow.db.backup
```

## CI/CD Integration

### GitHub Actions

See `.github/workflows/docker.yml` for automated Docker builds and tests.

### GitLab CI

```yaml
build:
  stage: build
  script:
    - docker build -t stock-prediction-api:$CI_COMMIT_SHA .
    - docker push stock-prediction-api:$CI_COMMIT_SHA
```

## Support

For issues and questions:
- Check logs: `docker-compose logs -f`
- Health endpoint: `http://localhost:8000/api/v1/health`
- MLflow UI: `http://localhost:5000`
- API docs: `http://localhost:8000/docs`
