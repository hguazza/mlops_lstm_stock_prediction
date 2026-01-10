# Ambiente de Pré-Produção - Guia Rápido

## 🚀 Início Rápido

### 1. Subir o Ambiente
```bash
docker compose -p pre-prod -f docker-compose.prod.yml --env-file .env.pre-prod up -d --build
```

### 2. Criar Usuário Admin (primeira vez apenas)
```bash
curl -X POST "http://localhost:8001/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@preprod.com", "password": "admin-password-123", "full_name": "PreProd Admin"}'
```

### 3. Executar Testes Automatizados
```bash
bash scripts/test-pre-prod.sh
```

### 4. Testar Persistência (após restart)
```bash
# Restart containers
docker compose -p pre-prod -f docker-compose.prod.yml restart

# Aguardar containers ficarem saudáveis
sleep 20

# Executar teste de persistência
bash scripts/test-pre-prod.sh --check-persistence
```

## 🔍 Endpoints Testados

| Endpoint | Método | Descrição | Status |
|----------|--------|-----------|--------|
| `/api/v1/health` | GET | Health check | ✅ |
| `/metrics` | GET | Prometheus metrics | ✅ |
| `/api/v1/auth/register` | POST | Registro de usuário | ✅ |
| `/api/v1/auth/login` | POST | Login JWT | ✅ |
| `/api/v1/multivariate/train-predict` | POST | Treinamento multivariate | ✅ |
| `/api/v1/models` | GET | Listar modelos | ✅ |
| `/api/v1/models/{symbol}/latest` | GET | Detalhes do modelo | ✅ |

## 📊 Portas

- **API**: http://localhost:8001
- **MLflow UI**: http://localhost:5002
- **PostgreSQL**: localhost:5432 (não exposta)

## 🛠️ Comandos Úteis

### Ver Logs
```bash
# API
docker logs stock-prediction-api -f

# MLflow Server  
docker logs mlflow-server -f

# PostgreSQL
docker logs stock-prediction-postgres -f
```

### Status dos Containers
```bash
docker ps
```

### Parar Ambiente
```bash
docker compose -p pre-prod -f docker-compose.prod.yml down
```

### Limpar Tudo (incluindo volumes)
```bash
docker compose -p pre-prod -f docker-compose.prod.yml down -v
```

### Acessar Container
```bash
# API
docker exec -it stock-prediction-api bash

# PostgreSQL
docker exec -it stock-prediction-postgres psql -U mlops_user -d stock_prediction
```

## 🧪 Exemplo de Teste Manual

### 1. Health Check
```bash
curl http://localhost:8001/api/v1/health
```

### 2. Login
```bash
curl -X POST "http://localhost:8001/api/v1/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@preprod.com", "password": "admin-password-123"}' \
  | jq -r '.access_token'
```

### 3. Treinar Modelo Multivariate
```bash
TOKEN="seu_token_aqui"

curl -X POST "http://localhost:8001/api/v1/multivariate/train-predict" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input_tickers": ["AAPL", "MSFT", "GOOG", "AMZN"],
    "target_ticker": "NVDA",
    "lookback": 60,
    "forecast_horizon": 5,
    "period": "1y"
  }'
```

### 4. Listar Modelos
```bash
curl -X GET "http://localhost:8001/api/v1/models" \
  -H "Authorization: Bearer $TOKEN"
```

### 5. Ver Métricas Prometheus
```bash
curl http://localhost:8001/metrics
```

## 💾 Persistência Verificada

Os seguintes dados persistem após restart dos containers:

- ✅ Usuários e autenticação (PostgreSQL)
- ✅ MLflow runs e experiments (PostgreSQL)
- ✅ MLflow model registry (PostgreSQL)
- ✅ Artefatos dos modelos (volume Docker)
- ✅ Métricas de treinamento (PostgreSQL via MLflow)

## 🔧 Troubleshooting

### Container reiniciando constantemente
```bash
# Ver logs
docker logs stock-prediction-api --tail 100

# Verificar healthcheck
docker inspect stock-prediction-api | jq '.[0].State.Health'
```

### MLflow não conecta
```bash
# Verificar se MLflow server está rodando
curl http://localhost:5002/health

# Ver logs do MLflow
docker logs mlflow-server --tail 50
```

### PostgreSQL não inicializa
```bash
# Ver logs
docker logs stock-prediction-postgres

# Verificar se databases foram criadas
docker exec stock-prediction-postgres psql -U mlops_user -c "\l"
```

### Limpar e recomeçar
```bash
# Parar e remover tudo
docker compose -p pre-prod -f docker-compose.prod.yml down -v

# Subir novamente
docker compose -p pre-prod -f docker-compose.prod.yml --env-file .env.pre-prod up -d --build

# Registrar usuário
curl -X POST "http://localhost:8001/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@preprod.com", "password": "admin-password-123", "full_name": "PreProd Admin"}'
```

## 📝 Credenciais Pré-Produção

**⚠️ ATENÇÃO: Estas credenciais são APENAS para pré-produção!**

```
Admin Email: admin@preprod.com
Admin Password: admin-password-123

PostgreSQL User: mlops_user  
PostgreSQL Password: pre-prod-secure-password
PostgreSQL DBs: stock_prediction, mlflow

JWT Secret: pre-prod-secret-key-for-testing-only-12345
```

## 🎯 Próximos Passos para Produção Real

1. **Segurança**:
   - Gerar novos secrets com `openssl rand -hex 32`
   - Usar senhas fortes
   - Configurar SSL/TLS
   - Não expor PostgreSQL

2. **Monitoramento**:
   - Configurar Prometheus
   - Setup Grafana dashboards
   - Alertas via Alertmanager

3. **Backup**:
   - Backup automático do PostgreSQL
   - Backup dos volumes MLflow artifacts
   - Testar restore procedures

4. **Escalabilidade**:
   - Considerar múltiplas réplicas da API
   - Load balancer
   - Cache (Redis)

5. **CI/CD**:
   - Pipeline de deploy automatizado
   - Testes de integração
   - Rollback automático

---

**Status**: ✅ Ambiente totalmente funcional e testado
**Última atualização**: 2026-01-10
