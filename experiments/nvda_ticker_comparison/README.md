# NVDA Ticker Comparison Experiment

Systematic comparison of different ticker combinations as input features for NVDA stock prediction.

## Quick Start

```bash
# 1. Ensure services are running
make docker-run

# 2. Wait for services (45s)
sleep 45

# 3. Run all experiments
cd experiments/nvda_ticker_comparison
chmod +x run_experiments.sh
./run_experiments.sh

# 4. Generate comparison report
python compare_results.py

# 5. View results
cat comparison_report.md
```

## Experiment Groups

1. **Tech Giants:** MSFT, AAPL, GOOGL, AMZN
2. **Semiconductors:** AMD, INTC, TSM, QCOM
3. **Mixed Sectors:** JPM, XOM, DIS, BA
4. **Cloud/AI:** MSFT, GOOGL, AMZN, META
5. **Gaming:** AMD, ATVI, EA, TTWO
6. **Automotive:** TSLA, F, GM, TM
7. **FinTech:** V, MA, PYPL, SQ
8. **DataCenter:** DELL, HPQ, CSCO, IBM

## Results Location

- **JSON Results:** `./results/*.json`
- **Comparison Report:** `./comparison_report.md`
- **MLflow UI:** http://localhost:5000

## Analysis

The comparison script generates:
- Performance rankings (MAE, RMSE)
- Statistical analysis
- Insights and recommendations
- Complete results table

## Duration

- Each experiment: ~2-3 minutes
- Total time: ~20-25 minutes
- Analysis: ~30 seconds
