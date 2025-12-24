# 🚀 Guia de Deploy em Produção

## 📋 Índice

1. [Visão Geral](#visão-geral)
2. [Problema do SQLite](#problema-do-sqlite)
3. [Arquitetura de Produção](#arquitetura-de-produção)
4. [Deploy Rápido](#deploy-rápido)
5. [Configuração Detalhada](#configuração-detalhada)
6. [Segurança](#segurança)
7. [Monitoramento](#monitoramento)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Visão Geral

Este guia explica como fazer deploy da API de predição de ações em **produção** usando PostgreSQL em vez de SQLite.

### Por que mudar do SQLite?

**Problema:** SQLite não suporta múltiplos processos escrevendo simultaneamente.

**Sintomas:**
- Erros: `Can't locate revision identified by '1bd49d398cd23'`
- Modelos não aparecem no Model Registry
- Database locks e timeouts

**Solução:** Use PostgreSQL para concorrência robusta.

---

## 🏗️ Arquitetura de Produção

```
┌─────────────────────────────────────────────────┐
│              Load Balancer / Nginx              │
│           (SSL/TLS Termination)                 │
└────────────┬────────────────────────┬───────────┘
             │                        │
             ▼                        ▼
   ┌─────────────────┐      ┌──────────────────┐
   │  API Service    │      │  MLflow Server   │
   │  (Port 8000)    │◄─────┤  (Port 5000)     │
   │  4 workers      │      │  Tracking + UI   │
   └────────┬────────┘      └────────┬─────────┘
            │                        │
            │    ┌──────────────────┐│
            └────►   PostgreSQL     ││
                 │   (Port 5432)    ││
                 │   - MLflow DB    ││
                 │   - Registry     ││
                 └──────────────────┘│
                          │
                 ┌────────▼──────────┐
                 │  Artifact Store   │
                 │  (Volume/S3)      │
                 └───────────────────┘
```

**Vantagens:**
- ✅ Zero conflitos de banco de dados
- ✅ Suporta múltiplos workers
- ✅ MLflow UI separado da API
- ✅ Escalável horizontalmente
- ✅ Backups simplificados

---

## ⚡ Deploy Rápido

### 1. Criar arquivo de ambiente

Crie `.env.production`:

```bash
# Database
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=seu_password_super_seguro_aqui
POSTGRES_DB=mlflow

# MLflow
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_ALLOWED_HOSTS=*

# API
API_WORKERS=4
LOG_LEVEL=info
```

### 2. Iniciar serviços

```bash
# Build e start com PostgreSQL
docker-compose -f docker-compose.prod.yml --env-file .env.production up -d

# Verificar status
docker-compose -f docker-compose.prod.yml ps

# Verificar logs
docker-compose -f docker-compose.prod.yml logs -f api
```

### 3. Verificar saúde

```bash
# API Health
curl http://localhost:8000/api/v1/health

# MLflow Health
curl http://localhost:5000/health

# Treinar modelo de teste
curl -X POST http://localhost:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "period": "1y", "config": {"epochs": 50}}'

# Verificar modelos registrados
curl http://localhost:8000/api/v1/models
```

---

## ⚙️ Configuração Detalhada

### Variáveis de Ambiente Críticas

#### Database (PostgreSQL)
```bash
POSTGRES_USER=mlflow              # Usuário do banco
POSTGRES_PASSWORD=***             # Password forte (min 16 chars)
POSTGRES_DB=mlflow                # Nome do database
```

#### MLflow
```bash
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_ALLOWED_HOSTS=api.seudominio.com,mlflow.seudominio.com
MLFLOW_EXPERIMENT_NAME=stock_prediction_lstm
```

#### API
```bash
API_WORKERS=4                     # 2-4 por CPU core
LOG_LEVEL=info                    # debug|info|warning|error
API_HOST=0.0.0.0
API_PORT=8000
```

### Recursos Recomendados

| Componente | CPU | RAM | Storage |
|------------|-----|-----|---------|
| API | 2 cores | 2GB | 10GB |
| MLflow Server | 1 core | 1GB | 5GB |
| PostgreSQL | 2 cores | 2GB | 20GB+ |

---

## 🔒 Segurança

### 1. Segredos e Senhas

**NUNCA** commite senhas no Git! Use:

```bash
# Gerar senha segura
openssl rand -base64 32

# Usar secrets manager (AWS, GCP, Azure)
export POSTGRES_PASSWORD=$(aws secretsmanager get-secret-value --secret-id prod/mlflow/db-password --query SecretString --output text)
```

### 2. Network Security

```yaml
# docker-compose.prod.yml
services:
  postgres:
    # NÃO expor porta externamente
    # ports:  # ← Comentado!
    #   - "5432:5432"
    networks:
      - stock-prediction-network  # Apenas internal
```

### 3. SSL/TLS

Use Nginx ou Traefik como reverse proxy:

```nginx
server {
    listen 443 ssl http2;
    server_name api.seudominio.com;

    ssl_certificate /etc/ssl/certs/api.crt;
    ssl_certificate_key /etc/ssl/private/api.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Rate Limiting

Adicione no Nginx:

```nginx
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

location /api/v1/train {
    limit_req zone=api_limit burst=5 nodelay;
    proxy_pass http://localhost:8000;
}
```

---

## 📊 Monitoramento

### 1. Health Checks

```bash
# Script de monitoramento
#!/bin/bash
HEALTH_URL="http://localhost:8000/api/v1/health"

response=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $response -eq 200 ]; then
    echo "✅ API is healthy"
else
    echo "❌ API is unhealthy (HTTP $response)"
    # Alertar via PagerDuty, Slack, etc
fi
```

### 2. Prometheus Metrics

```yaml
# docker-compose.prod.yml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'stock-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
```

### 3. Grafana Dashboard

```bash
docker-compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d
```

Acesse: http://localhost:3000

**Métricas importantes:**
- Request rate (req/s)
- Response time (p50, p95, p99)
- Error rate (%)
- Training duration
- Model registry size

---

## 🐳 Deploy em Cloud

### AWS (ECS/Fargate)

```bash
# 1. Build e push para ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
docker build -t stock-prediction-api .
docker tag stock-prediction-api:latest YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/stock-prediction-api:latest
docker push YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/stock-prediction-api:latest

# 2. RDS PostgreSQL
# Criar via console ou Terraform

# 3. Deploy no ECS
aws ecs create-cluster --cluster-name stock-prediction-cluster
```

### GCP (Cloud Run)

```bash
# 1. Build e push para GCR
gcloud builds submit --tag gcr.io/YOUR_PROJECT/stock-prediction-api

# 2. Cloud SQL PostgreSQL
gcloud sql instances create mlflow-db --tier=db-f1-micro --region=us-central1

# 3. Deploy no Cloud Run
gcloud run deploy stock-prediction-api \
  --image gcr.io/YOUR_PROJECT/stock-prediction-api \
  --platform managed \
  --region us-central1 \
  --add-cloudsql-instances YOUR_PROJECT:us-central1:mlflow-db
```

### Kubernetes (K8s)

```yaml
# k8s/deployment.yaml
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
        image: your-registry/stock-prediction-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "postgresql://mlflow:password@postgres:5432/mlflow"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

---

## 🔧 Troubleshooting

### Problema: "Can't locate revision"

**Causa:** Banco de dados corrompido ou migrações conflitantes.

**Solução:**
```bash
# Parar tudo
docker-compose -f docker-compose.prod.yml down -v

# Limpar volumes
docker volume rm mlops_lstm_stock_prediction_postgres-data

# Reiniciar
docker-compose -f docker-compose.prod.yml up -d
```

### Problema: API lenta

**Diagnóstico:**
```bash
# Verificar uso de recursos
docker stats

# Logs detalhados
docker-compose -f docker-compose.prod.yml logs -f api | grep "request_duration"
```

**Soluções:**
- Aumentar `API_WORKERS`
- Adicionar cache Redis
- Usar GPU para treinamento
- Escalar horizontalmente

### Problema: Out of Memory

**Solução:**
```yaml
# docker-compose.prod.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

---

## 📚 Referências

- [MLflow Deployment](https://mlflow.org/docs/latest/deployment.html)
- [PostgreSQL Tuning](https://www.postgresql.org/docs/current/performance-tips.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

---

## 📞 Suporte

Para issues específicos do projeto:
- GitHub Issues: [link-do-repo]
- Email: [seu-email]

**Última atualização:** Dezembro 2025
