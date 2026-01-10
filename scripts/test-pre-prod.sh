#!/bin/bash
# Pre-production automated test script

set -e

API_URL="http://localhost:8001/api/v1"
ADMIN_EMAIL="admin@preprod.com"
ADMIN_PASSWORD="admin-password-123"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=== Starting Pre-Production Tests ==="

# Helper function for JSON parsing
parse_json() {
    python3 -c "import sys, json; print(json.load(sys.stdin)$1)"
}

# 1. Health Check
echo -n "Checking Health Endpoint... "
HEALTH_RESPONSE=$(curl -s -f "$API_URL/health")
STATUS=$(echo "$HEALTH_RESPONSE" | parse_json "['status']")
if [ "$STATUS" == "healthy" ]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED: $HEALTH_RESPONSE${NC}"
    exit 1
fi

# 2. Metrics Check
echo -n "Checking Metrics Endpoint... "
METRICS_RESPONSE=$(curl -s -f "http://localhost:8001/metrics")
if [[ "$METRICS_RESPONSE" == *"http_requests_total"* ]]; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED: Metrics not found${NC}"
    exit 1
fi

# 3. JWT Flow - Login
echo -n "Authenticating... "
LOGIN_DATA="{\"email\": \"$ADMIN_EMAIL\", \"password\": \"$ADMIN_PASSWORD\"}"
LOGIN_RESPONSE=$(curl -s -X POST "$API_URL/auth/login" \
    -H "Content-Type: application/json" \
    -d "$LOGIN_DATA")

TOKEN=$(echo "$LOGIN_RESPONSE" | parse_json "['access_token']")

if [ -n "$TOKEN" ] && [ "$TOKEN" != "None" ]; then
    echo -e "${GREEN}Authenticated successfully${NC}"
else
    echo -e "${RED}Authentication failed: $LOGIN_RESPONSE${NC}"
    exit 1
fi

# 4. Multivariate Training
echo -n "Testing Multivariate Train-Predict (NVDA)... "
MULTIVARIATE_DATA='{
    "input_tickers": ["AAPL", "MSFT", "GOOG", "AMZN"],
    "target_ticker": "NVDA",
    "lookback": 60,
    "forecast_horizon": 5,
    "period": "1y"
}'

TRAIN_RESPONSE=$(curl -s -X POST "$API_URL/multivariate/train-predict" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "$MULTIVARIATE_DATA")

RUN_ID=$(echo "$TRAIN_RESPONSE" | parse_json "['mlflow_run_id']")
MODEL_ID=$(echo "$TRAIN_RESPONSE" | parse_json "['model_id']")

if [ -n "$RUN_ID" ] && [ "$RUN_ID" != "None" ]; then
    echo -e "${GREEN}Success! Run ID: $RUN_ID${NC}"
else
    echo -e "${RED}Training failed: $TRAIN_RESPONSE${NC}"
    exit 1
fi

# Store Run ID for persistence test
echo "$RUN_ID" > .last_run_id

echo "=== Initial Tests Passed! ==="
echo "To test persistence, run this script again with --check-persistence after restarting containers."

if [[ "$1" == "--check-persistence" ]]; then
    echo "=== Checking Persistence Across Restart ==="
    
    LAST_RUN_ID=$(cat .last_run_id)
    echo -n "Verifying Model in Registry for $LAST_RUN_ID... "
    
    MODELS_RESPONSE=$(curl -s -X GET "$API_URL/models" \
        -H "Authorization: Bearer $TOKEN")
    
    # Check if any model in the list has a latest_version (simple check for existence)
    if [[ "$MODELS_RESPONSE" == *"NVDA"* ]]; then
        echo -e "${GREEN}Model found in registry${NC}"
    else
        echo -e "${RED}Model NOT found in registry: $MODELS_RESPONSE${NC}"
        exit 1
    fi
    
    echo -n "Checking Model Details... "
    MODEL_DETAILS=$(curl -s -X GET "$API_URL/models/NVDA/latest" \
        -H "Authorization: Bearer $TOKEN")
    
    PERSISTED_RUN_ID=$(echo "$MODEL_DETAILS" | parse_json "['model_info']['mlflow_run_id']")
    
    if [ "$PERSISTED_RUN_ID" == "$LAST_RUN_ID" ]; then
        echo -e "${GREEN}Run ID matches! Persistence verified.${NC}"
    else
        echo -e "${RED}Run ID mismatch! Expected $LAST_RUN_ID, got $PERSISTED_RUN_ID${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}=== ALL PRE-PRODUCTION TESTS PASSED ===${NC}"
fi
