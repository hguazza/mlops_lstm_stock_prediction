# 🚀 Guia Rápido - Acessando a API pelo Navegador

**IP da API:** `http://34.61.75.148:8001`  
**MLflow UI:** `http://34.61.75.148:5002`

---

## 📖 Passo 1: Abrir a Documentação Interativa (Swagger UI)

Abra seu navegador e acesse:

```
http://34.61.75.148:8001/docs
```

Você verá uma interface interativa com todos os endpoints da API!

![Swagger UI](https://fastapi.tiangolo.com/img/index/index-01-swagger-ui-simple.png)

---

## 🏥 Passo 2: Testar o Health Check

1. Na página do Swagger, procure por **`GET /api/v1/health`**
2. Clique para expandir
3. Clique no botão **"Try it out"**
4. Clique em **"Execute"**

✅ Você deve ver uma resposta como:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "dependencies": {
    "mlflow": "connected",
    "yfinance": "accessible"
  }
}
```

---

## 🔐 Passo 3: Criar uma Conta de Usuário

### 3.1. Registrar Novo Usuário

1. Procure por **`POST /api/v1/auth/register`**
2. Clique em **"Try it out"**
3. Preencha o JSON de exemplo:

```json
{
  "email": "seu.email@example.com",
  "password": "SuaSenhaSegura123!",
  "full_name": "Seu Nome Completo"
}
```

4. Clique em **"Execute"**

✅ Resposta esperada (Status 201):
```json
{
  "id": 1,
  "email": "seu.email@example.com",
  "full_name": "Seu Nome Completo",
  "is_active": true,
  "created_at": "2026-01-10T..."
}
```

### 3.2. Fazer Login e Obter Token

1. Procure por **`POST /api/v1/auth/login`**
2. Clique em **"Try it out"**
3. Preencha com suas credenciais:

```json
{
  "username": "seu.email@example.com",
  "password": "SuaSenhaSegura123!"
}
```

4. Clique em **"Execute"**

✅ Copie o **`access_token`** da resposta:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 3.3. Autenticar no Swagger

1. No topo da página, clique no botão **"Authorize" 🔓**
2. Cole o token copiado no campo **Value** (sem "Bearer")
3. Clique em **"Authorize"**
4. Clique em **"Close"**

🎉 Agora você está autenticado e pode usar todos os endpoints!

---

## 📊 Passo 4: Treinar um Modelo

1. Procure por **`POST /api/v1/train`**
2. Clique em **"Try it out"**
3. Use este exemplo para treinar com dados da Apple:

```json
{
  "ticker": "AAPL",
  "start_date": "2020-01-01",
  "end_date": "2024-01-01",
  "model_config": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

4. Clique em **"Execute"**

⏳ Este processo pode levar alguns minutos...

✅ Resposta esperada:
```json
{
  "status": "success",
  "message": "Model trained successfully",
  "run_id": "abc123...",
  "metrics": {
    "test_rmse": 2.45,
    "test_mae": 1.89,
    "test_mape": 3.21
  }
}
```

📝 **Copie o `run_id`** para usar na previsão!

---

## 🔮 Passo 5: Fazer Previsões

1. Procure por **`POST /api/v1/predict`**
2. Clique em **"Try it out"**
3. Use o `run_id` do passo anterior:

```json
{
  "ticker": "AAPL",
  "run_id": "abc123...",
  "prediction_days": 30
}
```

4. Clique em **"Execute"**

✅ Você receberá previsões para os próximos 30 dias!

```json
{
  "ticker": "AAPL",
  "predictions": [
    {
      "date": "2024-01-02",
      "predicted_close": 185.23
    },
    {
      "date": "2024-01-03",
      "predicted_close": 186.45
    }
    // ... mais 28 dias
  ],
  "model_metrics": {
    "rmse": 2.45,
    "mae": 1.89
  }
}
```

---

## 🎨 Passo 6: Visualizar no MLflow

Abra em outra aba do navegador:

```
http://34.61.75.148:5002
```

No MLflow você pode:
- 📈 Ver métricas de treinamento (loss, accuracy, etc)
- 📊 Comparar diferentes experimentos
- 🔍 Inspecionar hiperparâmetros
- 📥 Baixar modelos treinados
- 📉 Ver gráficos de performance

### Como usar o MLflow:

1. **Ver Experimentos:** Na página inicial, você verá `production_experiment`
2. **Clicar no Experimento:** Mostra todas as execuções (runs)
3. **Selecionar um Run:** Clique no `run_id` para ver detalhes
4. **Ver Métricas:** Aba "Metrics" mostra gráficos de treino
5. **Ver Parâmetros:** Aba "Parameters" mostra configurações usadas

---

## 🔬 Passo 7: Modelo Multivariado (Avançado)

Para previsões mais sofisticadas usando múltiplas ações correlacionadas:

### 7.1. Treinar Modelo Multivariado

**`POST /api/v1/multivariate/train`**

```json
{
  "target_ticker": "NVDA",
  "feature_tickers": ["AMD", "INTC", "TSM"],
  "start_date": "2020-01-01",
  "end_date": "2024-01-01",
  "model_config": {
    "epochs": 50,
    "hidden_size": 100
  }
}
```

### 7.2. Prever com Modelo Multivariado

**`POST /api/v1/multivariate/predict`**

```json
{
  "target_ticker": "NVDA",
  "feature_tickers": ["AMD", "INTC", "TSM"],
  "run_id": "seu_run_id_aqui",
  "prediction_days": 30
}
```

---

## 🎯 Endpoints Úteis

| Endpoint | Descrição | Autenticação |
|----------|-----------|--------------|
| `GET /api/v1/health` | Verifica status da API | ❌ Não |
| `POST /api/v1/auth/register` | Criar conta | ❌ Não |
| `POST /api/v1/auth/login` | Fazer login | ❌ Não |
| `GET /api/v1/auth/me` | Ver seu perfil | ✅ Sim |
| `POST /api/v1/train` | Treinar modelo | ✅ Sim |
| `POST /api/v1/predict` | Fazer previsão | ✅ Sim |
| `GET /api/v1/models` | Listar modelos | ✅ Sim |
| `POST /api/v1/multivariate/train` | Treinar multivariado | ✅ Sim |
| `POST /api/v1/multivariate/predict` | Prever multivariado | ✅ Sim |

---

## 💡 Dicas

### ✅ Boas Práticas

- **Use períodos longos:** Pelo menos 2-3 anos de dados históricos
- **Tickers populares:** AAPL, GOOGL, MSFT, TSLA, NVDA funcionam bem
- **Epochs:** Comece com 50, aumente se necessário
- **Guarde o run_id:** Você precisa dele para fazer previsões

### ⚠️ Limitações

- API gratuita do yfinance tem rate limits
- Treinamento pode levar 2-10 minutos dependendo do período
- Token expira em 30 minutos (faça login novamente)

### 🐛 Problemas Comuns

**Erro 401 (Unauthorized):**
- Token expirou → Faça login novamente
- Esqueceu de autorizar → Clique no botão "Authorize"

**Erro 422 (Validation Error):**
- Formato de data incorreto → Use "YYYY-MM-DD"
- Ticker inválido → Verifique se o ticker existe no Yahoo Finance

**Erro 500 (Internal Server Error):**
- Ticker sem dados suficientes → Use período maior ou ticker diferente
- Servidor ocupado → Aguarde e tente novamente

---

## 📱 Acesso Direto via URL

Você também pode testar alguns endpoints direto na barra do navegador:

```
# Health Check (funciona sem autenticação)
http://34.61.75.148:8001/api/v1/health

# Ver documentação da API (OpenAPI spec)
http://34.61.75.148:8001/api/v1/openapi.json

# Documentação alternativa (ReDoc)
http://34.61.75.148:8001/redoc
```

---

## 🆘 Suporte

Se encontrar problemas:

1. **Verifique os logs:**
   ```bash
   gcloud compute ssh stock-prediction-vm --zone=us-central1-a \
     --command='cd ~/app && docker compose logs -f api'
   ```

2. **Teste a conexão:**
   ```bash
   curl http://34.61.75.148:8001/api/v1/health
   ```

3. **Reinicie os serviços:**
   ```bash
   gcloud compute ssh stock-prediction-vm --zone=us-central1-a \
     --command='cd ~/app && docker compose restart'
   ```

---

## 🎓 Próximos Passos

1. ✅ Testar diferentes tickers e períodos
2. ✅ Comparar modelos no MLflow
3. ✅ Experimentar o modelo multivariado
4. ✅ Ajustar hiperparâmetros para melhor performance
5. ✅ Integrar com seu próprio código Python/JavaScript

---

**🚀 Divirta-se explorando a API!**
