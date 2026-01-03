"""
Compare NVDA prediction experiments from different ticker combinations.
Analyzes MLflow runs and generates comprehensive comparison report.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_experiment_results(results_dir: str = "./results") -> List[Dict]:
    """Load all experiment results from JSON files."""
    results = []
    results_path = Path(results_dir)

    for json_file in sorted(results_path.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["experiment_name"] = json_file.stem
                results.append(data)
        except Exception as e:
            print(f"WARNING: Error loading {json_file}: {e}")

    return results


def create_comparison_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Create comparison DataFrame from experiment results."""
    comparison_data = []

    for result in results:
        row = {
            "Experiment": result["experiment_name"].replace("_", " ").title(),
            "Input Tickers": ", ".join(result["prediction"]["input_tickers"]),
            "MAE": result["training_metrics"]["mae"],
            "RMSE": result["training_metrics"]["rmse"],
            "Best Val Loss": result["training_metrics"]["best_val_loss"],
            "Final Train Loss": result["training_metrics"]["final_train_loss"],
            "Epochs Trained": result["training_metrics"]["epochs_trained"],
            "Training Time (s)": result["training_metrics"]["training_time_seconds"],
            "Predicted Return %": result["prediction"]["predicted_return_pct"],
            "Confidence Lower": result["prediction"]["confidence_interval"]["lower"],
            "Confidence Upper": result["prediction"]["confidence_interval"]["upper"],
            "MLflow Run ID": result.get("mlflow_run_id", "N/A"),
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Rank experiments by performance
    df["MAE Rank"] = df["MAE"].rank()
    df["RMSE Rank"] = df["RMSE"].rank()
    df["Combined Rank"] = (df["MAE Rank"] + df["RMSE Rank"]) / 2

    return df.sort_values("Combined Rank")


def generate_markdown_report(
    df: pd.DataFrame, output_file: str = "comparison_report.md"
):
    """Generate comprehensive markdown report."""

    report = f"""# NVDA Ticker Comparison Experiment Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Target Ticker:** NVDA
**Total Experiments:** {len(df)}

## Executive Summary

This experiment compares {len(df)} different combinations of ticker symbols as input features
for predicting NVDA stock returns using multivariate LSTM models.

### 🏆 Best Performing Combinations

#### By MAE (Mean Absolute Error)
"""

    # Top 3 by MAE
    top_mae = df.nsmallest(3, "MAE")
    for idx, row in top_mae.iterrows():
        report += f"""
**{int(row['MAE Rank'])}. {row['Experiment']}**
- Input Tickers: `{row['Input Tickers']}`
- MAE: `{row['MAE']:.4f}`
- RMSE: `{row['RMSE']:.4f}`
- Predicted Return: `{row['Predicted Return %']:.2f}%`
"""

    report += "\n#### By RMSE (Root Mean Square Error)\n"

    # Top 3 by RMSE
    top_rmse = df.nsmallest(3, "RMSE")
    for idx, row in top_rmse.iterrows():
        report += f"""
**{int(row['RMSE Rank'])}. {row['Experiment']}**
- Input Tickers: `{row['Input Tickers']}`
- MAE: `{row['MAE']:.4f}`
- RMSE: `{row['RMSE']:.4f}`
- Predicted Return: `{row['Predicted Return %']:.2f}%`
"""

    report += "\n## Complete Results Table\n\n"

    # Full results table (manual markdown)
    report += "| Experiment | Input Tickers | MAE | RMSE | Predicted Return % | Epochs Trained | Training Time (s) |\n"
    report += "|------------|---------------|-----|------|-------------------|----------------|-------------------|\n"
    for idx, row in df.iterrows():
        report += f"| {row['Experiment']} | {row['Input Tickers']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['Predicted Return %']:.2f}% | {row['Epochs Trained']} | {row['Training Time (s)']:.2f} |\n"

    report += "\n\n## Detailed Analysis\n\n"

    # Statistics
    report += f"""### Performance Statistics

- **Average MAE:** {df['MAE'].mean():.4f}
- **Average RMSE:** {df['RMSE'].mean():.4f}
- **Best MAE:** {df['MAE'].min():.4f} ({df.loc[df['MAE'].idxmin(), 'Experiment']})
- **Worst MAE:** {df['MAE'].max():.4f} ({df.loc[df['MAE'].idxmax(), 'Experiment']})
- **MAE Standard Deviation:** {df['MAE'].std():.4f}

### Training Efficiency

- **Average Training Time:** {df['Training Time (s)'].mean():.2f}s
- **Fastest Training:** {df['Training Time (s)'].min():.2f}s ({df.loc[df['Training Time (s)'].idxmin(), 'Experiment']})
- **Average Epochs:** {df['Epochs Trained'].mean():.1f}

## Insights & Recommendations

### Key Findings

1. **Best Overall Performance:** {df.iloc[0]['Experiment']}
   - Tickers: {df.iloc[0]['Input Tickers']}
   - Combined Rank: {df.iloc[0]['Combined Rank']:.2f}

2. **Performance Range:**
   - MAE range: {df['MAE'].min():.4f} to {df['MAE'].max():.4f}
   - Improvement of best over worst: {((df['MAE'].max() - df['MAE'].min()) / df['MAE'].max() * 100):.1f}%

3. **Sector Correlation:**
   - Semiconductor tickers (AMD, INTC, TSM, QCOM) show {'strong' if 'Semiconductors' in df.iloc[0]['Experiment'] else 'moderate'} correlation
   - Tech giants perform {'better' if 'Tech Giants' in df.iloc[:2]['Experiment'].values else 'comparably'}

### Recommendations

- ✅ **Recommended for Production:** {df.iloc[0]['Experiment']} ({df.iloc[0]['Input Tickers']})
- ⚡ **Best Speed/Performance Tradeoff:** {df.loc[df['Training Time (s)'].idxmin(), 'Experiment']}
- 📊 **For Further Testing:** Top 3 combinations should be tested with longer periods (2y, 5y)

## MLflow Tracking

All experiments are tracked in MLflow. View detailed metrics, model artifacts, and training curves:

```bash
# Open MLflow UI
open http://localhost:5000
```

### Run IDs

"""

    for idx, row in df.iterrows():
        report += f"- **{row['Experiment']}:** `{row['MLflow Run ID']}`\n"

    report += "\n## Appendix: Full Configuration\n\n"
    report += """
**Model Configuration:**
- Lookback: 60 days
- Forecast Horizon: 5 days
- Confidence Level: 95%
- Historical Period: 1 year
- Hidden Size 1: 128
- Hidden Size 2: 64
- Dropout: 0.3
- Attention: Enabled
- Learning Rate: 0.001
- Batch Size: 32
- Max Epochs: 100
- Early Stopping Patience: 15

**Features per Ticker:**
- Log Returns
- RSI (14-period)
- MACD Histogram
- Realized Volatility (20-day)
- Normalized Volume (20-day)

**Total Features:** 25 (5 features × 5 tickers, including target NVDA)
"""

    # Write report
    with open(output_file, "w") as f:
        f.write(report)

    print(f"SUCCESS: Report generated: {output_file}")


def main():
    """Main execution function."""
    print("Loading experiment results...")
    results = load_experiment_results()
    print(f"   Found {len(results)} experiments")

    if not results:
        print("ERROR: No results found. Run experiments first!")
        return

    print("\nCreating comparison DataFrame...")
    df = create_comparison_dataframe(results)

    print("\nPerformance Rankings:")
    print(df[["Experiment", "MAE", "RMSE", "Combined Rank"]].to_string(index=False))

    print("\nGenerating markdown report...")
    generate_markdown_report(df)

    print("\nTop 3 Recommendations:")
    for idx, row in df.head(3).iterrows():
        print(f"   {int(row['Combined Rank'])}. {row['Experiment']}")
        print(f"      Tickers: {row['Input Tickers']}")
        print(f"      MAE: {row['MAE']:.4f} | RMSE: {row['RMSE']:.4f}")
        print()


if __name__ == "__main__":
    main()
