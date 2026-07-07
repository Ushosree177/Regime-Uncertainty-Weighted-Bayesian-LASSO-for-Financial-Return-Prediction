"""
Statistical significance tests for repeated simulation results.

Purpose:

The repeated simulation showed that the proposed prior improves RMSE in the
Strong regime scenario. This script checks whether that improvement is
statistically meaningful.

Tests included:

1. Paired t-test on metric differences
2. 95 percent confidence interval for the mean difference
3. Cohen's d for paired differences

Comparisons:

    Proposed Prior vs Classical LASSO
    Proposed Prior vs Standard Bayesian LASSO

Metrics:

    RMSE
    MAE
    R2
    Direction Accuracy
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "repeated_simulation_study"
OUTPUT_DIR = DATA_DIR / "statistical_significance_tests"

METRICS_FILE = INPUT_DIR / "replication_level_model_metrics.csv"

PROPOSED_MODEL = "Proposed Prior"
BASELINES = ["Classical LASSO", "Standard Bayesian LASSO"]
METRICS = ["RMSE", "MAE", "R2", "Direction_Accuracy"]


def format_p_value(p_value):
    """Create publication-safe p-value text."""
    if not np.isfinite(p_value):
        return "NA"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def p_value_for_display(p_value):
    """Numeric p-value used only for rounded display tables."""
    if not np.isfinite(p_value):
        return np.nan
    return max(float(p_value), 0.001)


def paired_cohens_d(differences):
    """Cohen's d for paired samples."""
    std = differences.std(ddof=1)
    if std == 0:
        return np.inf
    return differences.mean() / std


def confidence_interval(differences, confidence=0.95):
    """Calculate a t-based confidence interval for paired differences."""
    n = len(differences)
    mean = differences.mean()
    standard_error = differences.std(ddof=1) / np.sqrt(n)

    if n <= 1 or standard_error == 0:
        return mean, mean

    critical_value = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    lower = mean - critical_value * standard_error
    upper = mean + critical_value * standard_error
    return lower, upper


def metric_direction(metric):
    """Return whether higher or lower is better for a metric."""
    if metric in ["RMSE", "MAE"]:
        return "Lower is better"
    return "Higher is better"


def calculate_difference(proposed_values, baseline_values, metric):
    """
    Calculate improvement so positive values always mean proposed is better.
    """
    if metric in ["RMSE", "MAE"]:
        return baseline_values - proposed_values

    return proposed_values - baseline_values


def run_tests(metrics):
    """Run paired tests for every scenario, baseline, and metric."""
    rows = []

    group_columns = ["Case", "Samples_Per_Regime"]

    for (case_name, samples_per_regime), group in metrics.groupby(group_columns):
        proposed = group[group["Model"] == PROPOSED_MODEL].sort_values("Replication")

        for baseline_name in BASELINES:
            baseline = group[group["Model"] == baseline_name].sort_values("Replication")

            merged = proposed.merge(
                baseline,
                on=["Case", "Samples_Per_Regime", "Replication"],
                suffixes=("_Proposed", "_Baseline"),
            )

            for metric in METRICS:
                proposed_values = merged[f"{metric}_Proposed"]
                baseline_values = merged[f"{metric}_Baseline"]
                differences = calculate_difference(proposed_values, baseline_values, metric)

                t_statistic, p_value = stats.ttest_rel(
                    proposed_values,
                    baseline_values,
                )

                # The t-test above tests raw equality. The sign depends on the metric.
                # The improvement statistics below are easier to interpret.
                ci_lower, ci_upper = confidence_interval(differences)
                effect_size = paired_cohens_d(differences)

                rows.append(
                    {
                        "Case": case_name,
                        "Samples_Per_Regime": samples_per_regime,
                        "Baseline": baseline_name,
                        "Metric": metric,
                        "Metric_Direction": metric_direction(metric),
                        "Proposed_Mean": proposed_values.mean(),
                        "Baseline_Mean": baseline_values.mean(),
                        "Improvement_Mean": differences.mean(),
                        "Improvement_95CI_Lower": ci_lower,
                        "Improvement_95CI_Upper": ci_upper,
                        "Paired_T_Statistic": t_statistic,
                        "Raw_P_Value": p_value,
                        "P_Value_Display": p_value_for_display(p_value),
                        "P_Value_Text": format_p_value(p_value),
                        "Cohens_D_Paired": effect_size,
                        "N_Replications": len(differences),
                        "Significant_0.05": p_value < 0.05,
                    }
                )

    return pd.DataFrame(rows)


def make_paper_summary(test_results):
    """Create a shorter table focused on RMSE and R2."""
    keep_metrics = ["RMSE", "R2", "Direction_Accuracy"]
    summary = test_results[test_results["Metric"].isin(keep_metrics)].copy()
    summary = summary[
        [
            "Case",
            "Samples_Per_Regime",
            "Baseline",
            "Metric",
            "Proposed_Mean",
            "Baseline_Mean",
            "Improvement_Mean",
            "Improvement_95CI_Lower",
            "Improvement_95CI_Upper",
            "P_Value_Display",
            "P_Value_Text",
            "Cohens_D_Paired",
            "Significant_0.05",
        ]
    ]
    return summary.sort_values(["Case", "Samples_Per_Regime", "Baseline", "Metric"])


def write_interpretation(test_results, paper_summary):
    """Write plain-English interpretation."""
    interpretation_file = OUTPUT_DIR / "significance_test_interpretation.txt"

    strong_rmse = paper_summary[
        (paper_summary["Case"] == "Strong") & (paper_summary["Metric"] == "RMSE")
    ]
    weak_rmse = paper_summary[
        (paper_summary["Case"] == "Weak") & (paper_summary["Metric"] == "RMSE")
    ]

    lines = []
    lines.append("STATISTICAL SIGNIFICANCE TEST INTERPRETATION")
    lines.append("")
    lines.append("Question:")
    lines.append("Are the repeated simulation improvements statistically meaningful?")
    lines.append("")
    lines.append("Strong scenario RMSE results:")
    lines.append(strong_rmse.round(6).to_string(index=False))
    lines.append("")
    lines.append("Weak scenario RMSE results:")
    lines.append(weak_rmse.round(6).to_string(index=False))
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("Positive improvement means the proposed prior is better.")
    lines.append("For RMSE and MAE, improvement = baseline - proposed.")
    lines.append("For R2 and direction accuracy, improvement = proposed - baseline.")
    lines.append("")
    lines.append("Research meaning:")
    lines.append(
        "If the Strong scenario has significant positive RMSE improvement, the "
        "simulation supports the claim that the proposed prior helps when regime "
        "structures are truly different."
    )
    lines.append(
        "If the Weak scenario has very small improvement, that supports the claim "
        "that the method should not be expected to dominate when regimes are not "
        "meaningfully different."
    )

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")
    return interpretation_file


def save_outputs(test_results):
    """Save test outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    full_file = OUTPUT_DIR / "paired_significance_tests_full.csv"
    summary_file = OUTPUT_DIR / "paired_significance_tests_summary.csv"
    paper_summary = make_paper_summary(test_results)
    interpretation_file = write_interpretation(test_results, paper_summary)

    test_results.round(8).to_csv(full_file, index=False)
    paper_summary.round(8).to_csv(summary_file, index=False)

    return {
        "full": full_file,
        "summary": summary_file,
        "interpretation": interpretation_file,
    }


def main():
    print("STATISTICAL SIGNIFICANCE TESTS")

    metrics = pd.read_csv(METRICS_FILE)
    test_results = run_tests(metrics)
    outputs = save_outputs(test_results)
    paper_summary = make_paper_summary(test_results)

    print("\nPAPER SUMMARY")
    print(paper_summary.round(6).to_string(index=False))

    print("\nSaved files:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
