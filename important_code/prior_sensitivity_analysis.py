"""
Step 1 scientific verification: prior sensitivity analysis.

We already proposed:

    lambda_jr = lambda_0 * exp(gamma * U_r) / (I_jr + epsilon)^delta

The first scientific check is simple:

    Do the shrinkage rankings remain stable when gamma and delta change?

This script tests:

    gamma in {0.5, 1.0, 2.0}
    delta in {0.5, 1.0, 2.0}

Outputs:

1. Heatmaps of log(lambda_jr)
2. Rank correlation table versus the baseline gamma=1, delta=1
3. Top low-shrinkage features by regime and parameter setting
4. Feature stability table showing how often each feature survives

No MCMC is used here. This is only a prior behavior check.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
FEASIBILITY_DIR = DATA_DIR / "prior_feasibility_check"
OUTPUT_DIR = DATA_DIR / "prior_sensitivity_analysis"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

UNCERTAINTY_FILE = FEASIBILITY_DIR / "table1_regime_uncertainty_scores.csv"
EVIDENCE_FILE = FEASIBILITY_DIR / "table2_feature_evidence_scores.csv"

REGIME_ORDER = ["Recovery", "Bull", "Bear", "Crisis"]

GAMMA_VALUES = [0.5, 1.0, 2.0]
DELTA_VALUES = [0.5, 1.0, 2.0]

LAMBDA_0 = 1.0
EPSILON = 0.001
TOP_K = 5


def load_inputs():
    """Load U_r and I_jr from the feasibility check."""
    uncertainty = pd.read_csv(UNCERTAINTY_FILE)
    evidence = pd.read_csv(EVIDENCE_FILE, index_col=0)

    evidence = evidence[[regime for regime in REGIME_ORDER if regime in evidence.columns]]
    uncertainty = uncertainty[uncertainty["Regime"].isin(evidence.columns)]

    return uncertainty, evidence


def calculate_lambda_matrix(uncertainty, evidence, gamma, delta):
    """Calculate lambda_jr for one gamma and delta setting."""
    u_values = uncertainty.set_index("Regime")["U_r"]
    lambda_matrix = evidence.copy().astype(float)

    for regime in lambda_matrix.columns:
        lambda_matrix[regime] = (
            LAMBDA_0
            * np.exp(gamma * u_values.loc[regime])
            / ((lambda_matrix[regime] + EPSILON) ** delta)
        )

    return lambda_matrix


def create_all_lambda_matrices(uncertainty, evidence):
    """Create lambda matrices for all gamma-delta combinations."""
    matrices = {}

    for gamma in GAMMA_VALUES:
        for delta in DELTA_VALUES:
            key = setting_name(gamma, delta)
            matrices[key] = calculate_lambda_matrix(uncertainty, evidence, gamma, delta)

    return matrices


def setting_name(gamma, delta):
    """Create a readable name for one parameter setting."""
    return f"gamma_{gamma}_delta_{delta}"


def plot_heatmap(matrix, title, output_file):
    """Save a heatmap of log(lambda_jr)."""
    log_matrix = np.log(matrix)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(log_matrix.values, aspect="auto", cmap="viridis")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(log_matrix.columns)))
    ax.set_xticklabels(log_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(log_matrix.index)))
    ax.set_yticklabels(log_matrix.index)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("log(lambda_jr)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()


def save_heatmaps(matrices):
    """Save heatmaps for all settings."""
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    heatmap_files = []
    for key, matrix in matrices.items():
        output_file = HEATMAP_DIR / f"{key}_heatmap.png"
        title = f"log(lambda_jr): {key}"
        plot_heatmap(matrix, title, output_file)
        heatmap_files.append(output_file)

    return heatmap_files


def calculate_rank_correlation(matrices):
    """
    Compare shrinkage rankings against the baseline gamma=1, delta=1.

    Lower lambda means the feature receives less shrinkage.
    Spearman correlation checks whether feature rankings are stable.
    """
    baseline_key = setting_name(1.0, 1.0)
    baseline = matrices[baseline_key]

    rows = []

    for key, matrix in matrices.items():
        gamma, delta = parse_setting_name(key)

        for regime in matrix.columns:
            baseline_rank = baseline[regime].rank(method="average")
            setting_rank = matrix[regime].rank(method="average")

            correlation = baseline_rank.corr(setting_rank, method="spearman")

            rows.append(
                {
                    "Setting": key,
                    "Gamma": gamma,
                    "Delta": delta,
                    "Regime": regime,
                    "Spearman_Rank_Correlation": correlation,
                }
            )

    return pd.DataFrame(rows)


def parse_setting_name(name):
    """Read gamma and delta from a setting name."""
    parts = name.split("_")
    gamma = float(parts[1])
    delta = float(parts[3])
    return gamma, delta


def create_top_feature_table(matrices):
    """List the lowest-shrinkage features for each regime and setting."""
    rows = []

    for key, matrix in matrices.items():
        gamma, delta = parse_setting_name(key)

        for regime in matrix.columns:
            top_features = matrix[regime].sort_values().head(TOP_K)

            for rank, (feature, lambda_value) in enumerate(top_features.items(), start=1):
                rows.append(
                    {
                        "Setting": key,
                        "Gamma": gamma,
                        "Delta": delta,
                        "Regime": regime,
                        "Rank": rank,
                        "Feature": feature,
                        "Lambda": lambda_value,
                    }
                )

    return pd.DataFrame(rows)


def create_survival_stability_table(top_feature_table):
    """
    Count how often each feature appears among the top low-shrinkage features.

    A stable feature should survive across many gamma-delta settings.
    """
    rows = []

    for regime in REGIME_ORDER:
        regime_rows = top_feature_table[top_feature_table["Regime"] == regime]
        counts = regime_rows["Feature"].value_counts()

        for feature, count in counts.items():
            rows.append(
                {
                    "Regime": regime,
                    "Feature": feature,
                    "Top_K_Appearances": count,
                    "Total_Settings": len(GAMMA_VALUES) * len(DELTA_VALUES),
                    "Stability_Rate": count / (len(GAMMA_VALUES) * len(DELTA_VALUES)),
                }
            )

    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["Regime", "Stability_Rate", "Top_K_Appearances", "Feature"],
        ascending=[True, False, False, True],
    )
    return table


def save_lambda_matrices(matrices):
    """Save every lambda matrix as a CSV file."""
    matrix_dir = OUTPUT_DIR / "lambda_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for key, matrix in matrices.items():
        output_file = matrix_dir / f"{key}.csv"
        matrix.round(6).to_csv(output_file)
        files.append(output_file)

    return files


def write_interpretation(rank_correlation, top_feature_table, stability_table):
    """Write a short interpretation of the sensitivity results."""
    interpretation_file = OUTPUT_DIR / "prior_sensitivity_interpretation.txt"

    mean_correlation = rank_correlation["Spearman_Rank_Correlation"].mean()
    min_correlation = rank_correlation["Spearman_Rank_Correlation"].min()

    lines = []
    lines.append("PRIOR SENSITIVITY ANALYSIS")
    lines.append("")
    lines.append("Question:")
    lines.append("Do shrinkage rankings remain stable when gamma and delta change?")
    lines.append("")
    lines.append(f"Gamma values tested: {GAMMA_VALUES}")
    lines.append(f"Delta values tested: {DELTA_VALUES}")
    lines.append("")
    lines.append(f"Average Spearman rank correlation: {mean_correlation:.6f}")
    lines.append(f"Minimum Spearman rank correlation: {min_correlation:.6f}")
    lines.append("")
    lines.append("Most stable low-shrinkage features by regime:")

    for regime in REGIME_ORDER:
        regime_stability = stability_table[stability_table["Regime"] == regime].head(5)
        lines.append("")
        lines.append(regime)
        lines.append(regime_stability.to_string(index=False))

    lines.append("")
    lines.append("Baseline top features for gamma=1, delta=1:")
    baseline_rows = top_feature_table[
        top_feature_table["Setting"] == setting_name(1.0, 1.0)
    ]
    lines.append(baseline_rows.to_string(index=False))
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("High rank correlation means the prior ranking is stable.")
    lines.append("High stability rate means a feature survives across many parameter settings.")

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")
    return interpretation_file


def save_outputs(matrices, rank_correlation, top_feature_table, stability_table):
    """Save all sensitivity-analysis outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    lambda_files = save_lambda_matrices(matrices)
    heatmap_files = save_heatmaps(matrices)

    rank_file = OUTPUT_DIR / "rank_correlation_vs_baseline.csv"
    top_file = OUTPUT_DIR / "top_low_shrinkage_features.csv"
    stability_file = OUTPUT_DIR / "feature_survival_stability.csv"

    rank_correlation.round(6).to_csv(rank_file, index=False)
    top_feature_table.round(6).to_csv(top_file, index=False)
    stability_table.round(6).to_csv(stability_file, index=False)

    interpretation_file = write_interpretation(
        rank_correlation,
        top_feature_table,
        stability_table,
    )

    return {
        "rank_correlation": rank_file,
        "top_features": top_file,
        "stability": stability_file,
        "interpretation": interpretation_file,
        "lambda_files": lambda_files,
        "heatmaps": heatmap_files,
    }


def main():
    print("PRIOR SENSITIVITY ANALYSIS")
    print("No Bayesian sampler is used here.")

    uncertainty, evidence = load_inputs()
    matrices = create_all_lambda_matrices(uncertainty, evidence)

    rank_correlation = calculate_rank_correlation(matrices)
    top_feature_table = create_top_feature_table(matrices)
    stability_table = create_survival_stability_table(top_feature_table)

    outputs = save_outputs(matrices, rank_correlation, top_feature_table, stability_table)

    print("\nRank correlation versus baseline gamma=1, delta=1")
    print(rank_correlation.round(6).to_string(index=False))

    print("\nFeature survival stability")
    print(stability_table.round(6).to_string(index=False))

    print("\nSaved main files:")
    print(outputs["rank_correlation"])
    print(outputs["top_features"])
    print(outputs["stability"])
    print(outputs["interpretation"])
    print(f"Heatmaps saved: {len(outputs['heatmaps'])}")


if __name__ == "__main__":
    main()
