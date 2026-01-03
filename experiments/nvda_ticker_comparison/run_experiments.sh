#!/bin/bash
set -e

echo "🚀 Starting NVDA Ticker Comparison Experiment"
echo "=============================================="

# Configuration
API_URL="http://localhost:8000/api/v1/multivariate/train-predict"
RESULTS_DIR="./results"
mkdir -p "$RESULTS_DIR"

# Function to run experiment
run_experiment() {
    local name=$1
    local tickers=$2
    local description=$3

    echo ""
    echo "📊 Testing: $name"
    echo "   Tickers: $tickers"
    echo "   Description: $description"

    curl -X POST "$API_URL" \
      -H "Content-Type: application/json" \
      -d "{
        \"input_tickers\": $tickers,
        \"target_ticker\": \"NVDA\",
        \"lookback\": 60,
        \"forecast_horizon\": 5,
        \"confidence_level\": 0.95,
        \"period\": \"1y\"
      }" \
      -o "$RESULTS_DIR/${name}.json" \
      --silent --show-error

    if [ $? -eq 0 ]; then
        echo "   ✅ Completed successfully"
        # Extract key metrics
        mae=$(cat "$RESULTS_DIR/${name}.json" | python -c "import sys, json; print(json.load(sys.stdin)['training_metrics']['mae'])" 2>/dev/null || echo "N/A")
        rmse=$(cat "$RESULTS_DIR/${name}.json" | python -c "import sys, json; print(json.load(sys.stdin)['training_metrics']['rmse'])" 2>/dev/null || echo "N/A")
        prediction=$(cat "$RESULTS_DIR/${name}.json" | python -c "import sys, json; print(json.load(sys.stdin)['prediction']['predicted_return_pct'])" 2>/dev/null || echo "N/A")
        echo "   📈 MAE: $mae | RMSE: $rmse | Prediction: $prediction%"
    else
        echo "   ❌ Failed"
    fi

    # Wait between requests
    sleep 2
}

# Group 1: Tech Giants (FAANG-like)
run_experiment "tech_giants" \
    '["MSFT", "AAPL", "GOOGL", "AMZN"]' \
    "Major technology companies with diverse portfolios"

# Group 2: Semiconductors (Direct competitors/partners)
run_experiment "semiconductors" \
    '["AMD", "INTC", "TSM", "QCOM"]' \
    "Semiconductor manufacturers - direct industry correlation"

# Group 3: Mixed Sectors (Diversified)
run_experiment "mixed_sectors" \
    '["JPM", "XOM", "DIS", "BA"]' \
    "Diverse sectors to test cross-industry correlation"

# Group 4: Cloud/AI Leaders
run_experiment "cloud_ai" \
    '["MSFT", "GOOGL", "AMZN", "META"]' \
    "Cloud computing and AI infrastructure leaders"

# Group 5: Gaming/Graphics Related
run_experiment "gaming" \
    '["AMD", "ATVI", "EA", "TTWO"]' \
    "Gaming industry - major GPU/graphics card consumers"

# Group 6: Automotive Tech (EV/Autonomous)
run_experiment "automotive" \
    '["TSLA", "F", "GM", "TM"]' \
    "Automotive industry using AI chips for autonomous driving"

# Group 7: FinTech/Payments
run_experiment "fintech" \
    '["V", "MA", "PYPL", "SQ"]' \
    "Financial technology companies"

# Group 8: DataCenter/Infrastructure
run_experiment "datacenter" \
    '["DELL", "HPQ", "CSCO", "IBM"]' \
    "Data center infrastructure providers - NVDA customers"

echo ""
echo "=============================================="
echo "✅ All experiments completed!"
echo "📁 Results saved in: $RESULTS_DIR/"
echo ""
echo "Next steps:"
echo "  1. Run comparison script: python compare_results.py"
echo "  2. Check MLflow UI: http://localhost:5000"
echo "  3. Review report: cat comparison_report.md"
