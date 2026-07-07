"""
Rolling-window validation for the real-data experiment.

Purpose:

The earlier real-data experiment used one fixed train-test split. This script
adds a walk-forward validation design, which is more appropriate for financial
forecasting.

For each test year:

    Train on all years before the test year
    Test on the selected year

Models compared:

1. Classical LASSO
2. Elastic Net
3. Adaptive LASSO
4. Regime-specific LASSO
5. Proposed Prior - Full

The output answers:

    Are the conclusions stable across time?
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robust_real_data_validation import (
    FEATURE_COLUMNS,
    REGIME_ORDER,
    TARGET_COLUMN,
    RANDOM_STATE,
    add_research_features,
    calculate_feature_evidence,
    calculate_lambda_matrix,
    calculate_regime_uncertainty,
    load_data,
    prepare_model_data,
    train_adaptive_lasso,
    train_elastic_net,
    train_lasso,
    train_regime_bayesian_model,
    train_regime_lasso,
)


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "rolling_window_validation"

START_TEST_YEAR = 2015
END_TEST_YEAR = 2025
MIN_TRAIN_ROWS = 20000


def make_yearly_windows(model_data):
    """Create expanding train and one-year test windows."""
    windows = []

    for test_year in range(START_TEST_YEAR, END_TEST_YEAR + 1):
        train_data = model_data[model_data["Date"].dt.year < test_year].copy()
        test_data = model_data[model_data["Date"].dt.year == test_year].copy()

        if len(train_data) < MIN_TRAIN_ROWS or test_data.empty:
            continue

        windows.append(
            {
                "Test_Year": test_year,
                "Train_Data": train_data,
                "Test_Data": test_data,
            }
        )

    return windows


def run_one_window(window):
    """Run all selected models for one test year."""
    test_year = window["Test_Year"]
    train_data = window["Train_Data"]
    test_data = window["Test_Data"]

    print(f"\nRolling window test year: {test_year}")
    print(f"Training rows: {len(train_data)}")
    print(f"Testing rows: {len(test_data)}")

    selected_rows = []
    result_rows = []

    result_rows.append(train_lasso(train_data, test_data, selected_rows))
    result_rows.append(train_elastic_net(train_data, test_data, selected_rows))
    result_rows.append(train_adaptive_lasso(train_data, test_data, selected_rows))
    result_rows.append(train_regime_lasso(train_data, test_data, selected_rows))

    uncertainty = calculate_regime_uncertainty(train_data)
    evidence = calculate_feature_evidence(train_data)
    full_lambda = calculate_lambda_matrix(uncertainty, evidence, "full")
    result_rows.append(
        train_regime_bayesian_model(
            train_data,
            test_data,
            selected_rows,
            full_lambda,
            "Proposed Prior - Full",
        )
    )

    for row in result_rows:
        row["Test_Year"] = test_year
        row["Train_Rows"] = len(train_data)
        row["Window_Test_Rows"] = len(test_data)

    return result_rows


def summarize_results(results):
    """Create mean and standard deviation by model."""
    summary = (
        results.groupby("Model")
        .agg(
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            MAE_Mean=("MAE", "mean"),
            MAE_Std=("MAE", "std"),
            R2_Mean=("R2", "mean"),
            R2_Std=("R2", "std"),
            Direction_Accuracy_Mean=("Direction_Accuracy", "mean"),
            Direction_Accuracy_Std=("Direction_Accuracy", "std"),
            Windows=("Test_Year", "nunique"),
        )
        .reset_index()
    )

    summary = summary.sort_values("RMSE_Mean").reset_index(drop=True)
    summary["Rank"] = np.arange(1, len(summary) + 1)
    return summary


def plot_rmse_by_year(results):
    """Create a time-series plot of RMSE by model."""
    output_file = OUTPUT_DIR / "rolling_window_rmse_by_year.png"

    plt.figure(figsize=(11, 6))

    for model, group in results.groupby("Model"):
        group = group.sort_values("Test_Year")
        plt.plot(group["Test_Year"], group["RMSE"], marker="o", linewidth=1.6, label=model)

    plt.title("Rolling-Window RMSE by Test Year")
    plt.xlabel("Test Year")
    plt.ylabel("RMSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    return output_file


def write_interpretation(results, summary):
    """Write a concise interpretation file."""
    interpretation_file = OUTPUT_DIR / "rolling_window_interpretation.txt"

    best_model = summary.iloc[0]["Model"]
    proposed = summary[summary["Model"] == "Proposed Prior - Full"]

    lines = []
    lines.append("ROLLING-WINDOW VALIDATION INTERPRETATION")
    lines.append("")
    lines.append(f"Test years: {START_TEST_YEAR} to {END_TEST_YEAR}")
    lines.append("")
    lines.append("Average performance across windows:")
    lines.append(summary.round(6).to_string(index=False))
    lines.append("")
    lines.append(f"Best average RMSE model: {best_model}")

    if not proposed.empty:
        proposed_row = proposed.iloc[0]
        lines.append(
            f"Proposed prior average RMSE: {proposed_row['RMSE_Mean']:.6f} "
            f"+/- {proposed_row['RMSE_Std']:.6f}"
        )

    lines.append("")
    lines.append("Research meaning:")
    lines.append(
        "Rolling-window validation checks whether the fixed train-test result is "
        "stable across different market periods."
    )
    lines.append(
        "If the proposed prior remains close to simpler sparse baselines, but does "
        "not dominate, this supports the earlier conclusion that real data has "
        "weaker regime-dependent structure than the strong simulation scenario."
    )

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")
    return interpretation_file


def save_outputs(results):
    """Save rolling-window outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary = summarize_results(results)

    results_file = OUTPUT_DIR / "rolling_window_results.csv"
    summary_file = OUTPUT_DIR / "rolling_window_summary.csv"
    plot_file = plot_rmse_by_year(results)
    interpretation_file = write_interpretation(results, summary)

    results.round(6).to_csv(results_file, index=False)
    summary.round(6).to_csv(summary_file, index=False)

    return {
        "results": results_file,
        "summary": summary_file,
        "plot": plot_file,
        "interpretation": interpretation_file,
    }


def main():
    print("ROLLING-WINDOW VALIDATION")

    raw_data = load_data()
    data_with_features = add_research_features(raw_data)
    model_data = prepare_model_data(data_with_features)

    windows = make_yearly_windows(model_data)
    print(f"Number of rolling windows: {len(windows)}")

    all_rows = []
    for window in windows:
        all_rows.extend(run_one_window(window))

    results = pd.DataFrame(all_rows)
    outputs = save_outputs(results)
    summary = summarize_results(results)

    print("\nROLLING-WINDOW SUMMARY")
    print(summary.round(6).to_string(index=False))

    print("\nSaved files:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
