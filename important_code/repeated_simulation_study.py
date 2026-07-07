"""
Repeated simulation study.

Purpose:

The first simulation study showed that the proposed prior helps strongly when
regime-specific feature structures are strong. But one simulation run is not
enough for a paper.

This script repeats the simulation many times with different random seeds and
reports average performance and standard deviation.

By default, it uses 10 replications so it can run on a normal laptop. For the
final paper, change N_REPLICATIONS to 30.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from simulation_study import run_one_scenario


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "repeated_simulation_study"

RANDOM_STATE = 42
N_REPLICATIONS = 10

SCENARIOS = [
    ("Weak", 5000),
    ("Weak", 10000),
    ("Strong", 5000),
    ("Strong", 10000),
]


def run_repeated_simulations():
    """Run every scenario for many random seeds."""
    all_metric_rows = []
    all_recovery_rows = []
    all_selected_rows = []

    for replication in range(1, N_REPLICATIONS + 1):
        print(f"\n========== Replication {replication} of {N_REPLICATIONS} ==========")

        for scenario_number, (case_name, samples_per_regime) in enumerate(SCENARIOS):
            seed = RANDOM_STATE + replication * 100 + scenario_number * 10

            metric_rows, recovery_rows, selected_rows = run_one_scenario(
                case_name,
                samples_per_regime,
                seed,
            )

            for row in metric_rows:
                row["Replication"] = replication
                all_metric_rows.append(row)

            for row in recovery_rows:
                row["Replication"] = replication
                all_recovery_rows.append(row)

            for row in selected_rows:
                row["Replication"] = replication
                all_selected_rows.append(row)

    metrics = pd.DataFrame(all_metric_rows)
    recovery = pd.DataFrame(all_recovery_rows)
    selected = pd.DataFrame(all_selected_rows)

    return metrics, recovery, selected


def summarize_metrics(metrics):
    """Create mean and standard deviation table for prediction metrics."""
    summary = (
        metrics.groupby(["Case", "Samples_Per_Regime", "Model"])
        .agg(
            RMSE_Mean=("RMSE", "mean"),
            RMSE_Std=("RMSE", "std"),
            MAE_Mean=("MAE", "mean"),
            MAE_Std=("MAE", "std"),
            R2_Mean=("R2", "mean"),
            R2_Std=("R2", "std"),
            Direction_Accuracy_Mean=("Direction_Accuracy", "mean"),
            Direction_Accuracy_Std=("Direction_Accuracy", "std"),
        )
        .reset_index()
    )

    summary = summary.sort_values(["Case", "Samples_Per_Regime", "RMSE_Mean"])
    return summary


def summarize_recovery(recovery):
    """Create mean and standard deviation table for feature recovery."""
    summary = (
        recovery.groupby(["Case", "Samples_Per_Regime", "Model"])
        .agg(
            Precision_Mean=("Precision", "mean"),
            Precision_Std=("Precision", "std"),
            Recall_Mean=("Recall", "mean"),
            Recall_Std=("Recall", "std"),
            F1_Mean=("F1", "mean"),
            F1_Std=("F1", "std"),
            Selected_Count_Mean=("Selected_Count", "mean"),
            Selected_Count_Std=("Selected_Count", "std"),
        )
        .reset_index()
    )

    summary = summary.sort_values(["Case", "Samples_Per_Regime", "F1_Mean"], ascending=[True, True, False])
    return summary


def summarize_improvement(metrics):
    """Compare proposed prior against Classical LASSO and Standard Bayesian LASSO."""
    rows = []

    for (case_name, samples_per_regime, replication), group in metrics.groupby(
        ["Case", "Samples_Per_Regime", "Replication"]
    ):
        proposed = group[group["Model"] == "Proposed Prior"].iloc[0]

        for baseline_name in ["Classical LASSO", "Standard Bayesian LASSO"]:
            baseline = group[group["Model"] == baseline_name].iloc[0]

            rows.append(
                {
                    "Case": case_name,
                    "Samples_Per_Regime": samples_per_regime,
                    "Replication": replication,
                    "Baseline": baseline_name,
                    "RMSE_Improvement": baseline["RMSE"] - proposed["RMSE"],
                    "RMSE_Improvement_Percent": 100 * (baseline["RMSE"] - proposed["RMSE"]) / baseline["RMSE"],
                    "R2_Improvement": proposed["R2"] - baseline["R2"],
                    "Direction_Accuracy_Improvement": proposed["Direction_Accuracy"] - baseline["Direction_Accuracy"],
                }
            )

    improvement = pd.DataFrame(rows)

    summary = (
        improvement.groupby(["Case", "Samples_Per_Regime", "Baseline"])
        .agg(
            RMSE_Improvement_Mean=("RMSE_Improvement", "mean"),
            RMSE_Improvement_Std=("RMSE_Improvement", "std"),
            RMSE_Improvement_Percent_Mean=("RMSE_Improvement_Percent", "mean"),
            R2_Improvement_Mean=("R2_Improvement", "mean"),
            Direction_Accuracy_Improvement_Mean=("Direction_Accuracy_Improvement", "mean"),
        )
        .reset_index()
    )

    return improvement, summary


def summarize_selected_features(selected):
    """Calculate how often each feature is selected by each model."""
    total_replications = selected[["Case", "Samples_Per_Regime", "Model", "Replication"]].drop_duplicates()
    total_replications = (
        total_replications.groupby(["Case", "Samples_Per_Regime", "Model"])
        .size()
        .reset_index(name="Total_Replications")
    )

    feature_counts = (
        selected.groupby(["Case", "Samples_Per_Regime", "Model", "Regime", "Feature"])
        .size()
        .reset_index(name="Selected_Count")
    )

    feature_counts = feature_counts.merge(
        total_replications,
        on=["Case", "Samples_Per_Regime", "Model"],
        how="left",
    )
    feature_counts["Selection_Frequency"] = (
        feature_counts["Selected_Count"] / feature_counts["Total_Replications"]
    )

    return feature_counts.sort_values(
        ["Case", "Samples_Per_Regime", "Model", "Regime", "Selection_Frequency"],
        ascending=[True, True, True, True, False],
    )


def write_interpretation(metric_summary, recovery_summary, improvement_summary):
    """Write a paper-style interpretation of the repeated simulation."""
    interpretation_file = OUTPUT_DIR / "repeated_simulation_interpretation.txt"

    lines = []
    lines.append("REPEATED SIMULATION STUDY INTERPRETATION")
    lines.append("")
    lines.append(f"Number of replications: {N_REPLICATIONS}")
    lines.append("")
    lines.append("Prediction summary:")
    lines.append(metric_summary.round(6).to_string(index=False))
    lines.append("")
    lines.append("Feature recovery summary:")
    lines.append(recovery_summary.round(6).to_string(index=False))
    lines.append("")
    lines.append("Improvement of proposed prior over baselines:")
    lines.append(improvement_summary.round(6).to_string(index=False))
    lines.append("")
    lines.append("Research meaning:")
    lines.append(
        "Repeated simulations reduce the chance that the earlier simulation result "
        "was caused by one lucky random seed."
    )
    lines.append(
        "If the proposed prior has positive average RMSE improvement in the Strong "
        "case, it supports the claim that adaptive sparsity helps when regimes "
        "really have different active features."
    )

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")
    return interpretation_file


def save_outputs(metrics, recovery, selected):
    """Save all repeated simulation outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    metric_summary = summarize_metrics(metrics)
    recovery_summary = summarize_recovery(recovery)
    improvement, improvement_summary = summarize_improvement(metrics)
    selection_frequency = summarize_selected_features(selected)

    raw_metrics_file = OUTPUT_DIR / "replication_level_model_metrics.csv"
    raw_recovery_file = OUTPUT_DIR / "replication_level_feature_recovery.csv"
    metric_summary_file = OUTPUT_DIR / "average_model_metrics.csv"
    recovery_summary_file = OUTPUT_DIR / "average_feature_recovery.csv"
    improvement_file = OUTPUT_DIR / "replication_level_improvements.csv"
    improvement_summary_file = OUTPUT_DIR / "average_improvements.csv"
    selection_frequency_file = OUTPUT_DIR / "feature_selection_frequency.csv"

    metrics.round(6).to_csv(raw_metrics_file, index=False)
    recovery.round(6).to_csv(raw_recovery_file, index=False)
    metric_summary.round(6).to_csv(metric_summary_file, index=False)
    recovery_summary.round(6).to_csv(recovery_summary_file, index=False)
    improvement.round(6).to_csv(improvement_file, index=False)
    improvement_summary.round(6).to_csv(improvement_summary_file, index=False)
    selection_frequency.round(6).to_csv(selection_frequency_file, index=False)

    interpretation_file = write_interpretation(
        metric_summary,
        recovery_summary,
        improvement_summary,
    )

    return {
        "raw_metrics": raw_metrics_file,
        "raw_recovery": raw_recovery_file,
        "metric_summary": metric_summary_file,
        "recovery_summary": recovery_summary_file,
        "improvement": improvement_file,
        "improvement_summary": improvement_summary_file,
        "selection_frequency": selection_frequency_file,
        "interpretation": interpretation_file,
    }


def main():
    print("REPEATED SIMULATION STUDY")
    print(f"Replications: {N_REPLICATIONS}")

    metrics, recovery, selected = run_repeated_simulations()
    outputs = save_outputs(metrics, recovery, selected)

    metric_summary = summarize_metrics(metrics)
    recovery_summary = summarize_recovery(recovery)
    _, improvement_summary = summarize_improvement(metrics)

    print("\nAVERAGE MODEL METRICS")
    print(metric_summary.round(6).to_string(index=False))

    print("\nAVERAGE FEATURE RECOVERY")
    print(recovery_summary.round(6).to_string(index=False))

    print("\nAVERAGE IMPROVEMENTS")
    print(improvement_summary.round(6).to_string(index=False))

    print("\nSaved files:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
