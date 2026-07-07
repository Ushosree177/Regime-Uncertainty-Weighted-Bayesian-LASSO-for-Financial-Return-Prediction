"""
Feasibility check for the Regime-Uncertainty Weighted Bayesian LASSO prior.

This is not a Bayesian sampler.
This script only checks whether the proposed prior produces sensible shrinkage
values before we spend time implementing the full model.

It creates three tables:

1. Regime uncertainty scores:
       U_r = z(Volatility_r) + z(1 - Persistence_r)

2. Feature evidence scores:
       I_jr = normalized average feature importance for feature j in regime r

3. Shrinkage matrix:
       lambda_jr = lambda_0 * exp(gamma * U_r) / (I_jr + epsilon)^delta

Smaller lambda means weaker shrinkage, so the feature is more likely to survive.
Larger lambda means stronger shrinkage, so the feature is more likely to be removed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
REGIME_ANALYSIS_DIR = DATA_DIR / "regime_deep_analysis"
FEATURE_IMPORTANCE_DIR = DATA_DIR / "regime_feature_importance"
OUTPUT_DIR = DATA_DIR / "prior_feasibility_check"

REGIME_STATS_FILE = REGIME_ANALYSIS_DIR / "day3_regime_statistics.csv"
TRANSITION_MATRIX_FILE = REGIME_ANALYSIS_DIR / "day2_transition_matrix.csv"
FEATURE_IMPORTANCE_FILE = FEATURE_IMPORTANCE_DIR / "all_feature_importance_long.csv"

REGIME_ORDER = ["Recovery", "Bull", "Bear", "Crisis"]

LAMBDA_0 = 1.0
GAMMA = 1.0
DELTA = 1.0
EPSILON = 0.001


def z_score(series):
    """Calculate z-score using population standard deviation."""
    return (series - series.mean()) / series.std(ddof=0)


def load_regime_volatility():
    """Read Daily_Return standard deviation from the saved regime statistics."""
    stats = pd.read_csv(REGIME_STATS_FILE, header=[0, 1], index_col=0)

    # Remove the extra label row that appears because the CSV has a multi-row header.
    stats = stats[stats.index.notna()]
    stats = stats[stats.index != "Regime"]

    volatility = stats[("Daily_Return", "std")].astype(float)
    volatility.name = "Volatility"
    return volatility


def load_regime_persistence():
    """Read diagonal values from the transition matrix."""
    transition_matrix = pd.read_csv(TRANSITION_MATRIX_FILE, index_col=0)

    persistence_values = {}
    for regime in transition_matrix.index:
        persistence_values[regime] = float(transition_matrix.loc[regime, regime])

    persistence = pd.Series(persistence_values, name="Persistence")
    return persistence


def calculate_regime_uncertainty():
    """Create Table 1: regime uncertainty scores."""
    volatility = load_regime_volatility()
    persistence = load_regime_persistence()

    table = pd.concat([volatility, persistence], axis=1)
    table = table.loc[[regime for regime in REGIME_ORDER if regime in table.index]]

    table["Instability"] = 1 - table["Persistence"]
    table["Z_Volatility"] = z_score(table["Volatility"])
    table["Z_Instability"] = z_score(table["Instability"])
    table["U_r"] = table["Z_Volatility"] + table["Z_Instability"]

    return table.reset_index().rename(columns={"index": "Regime"})


def calculate_feature_evidence():
    """Create Table 2: normalized feature evidence scores I_jr."""
    importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)

    averaged = (
        importance.groupby(["Feature", "Regime"])["Normalized_Importance"]
        .mean()
        .reset_index()
    )

    min_value = averaged["Normalized_Importance"].min()
    max_value = averaged["Normalized_Importance"].max()

    averaged["I_jr"] = (
        (averaged["Normalized_Importance"] - min_value)
        / (max_value - min_value)
    )

    evidence_matrix = averaged.pivot(
        index="Feature",
        columns="Regime",
        values="I_jr",
    )

    existing_regimes = [regime for regime in REGIME_ORDER if regime in evidence_matrix.columns]
    evidence_matrix = evidence_matrix[existing_regimes]
    evidence_matrix = evidence_matrix.round(6)

    return averaged, evidence_matrix


def calculate_lambda_matrix(uncertainty_table, evidence_matrix):
    """Create Table 3: lambda_jr shrinkage values."""
    uncertainty = uncertainty_table.set_index("Regime")["U_r"]

    lambda_matrix = evidence_matrix.copy().astype(float)

    for regime in lambda_matrix.columns:
        regime_uncertainty = uncertainty.loc[regime]
        evidence = lambda_matrix[regime]

        lambda_matrix[regime] = (
            LAMBDA_0
            * np.exp(GAMMA * regime_uncertainty)
            / ((evidence + EPSILON) ** DELTA)
        )

    return lambda_matrix.round(6)


def create_interpretation(uncertainty_table, evidence_matrix, lambda_matrix):
    """Write a simple explanation of what the tables mean."""
    report = []

    report.append("PRIOR FEASIBILITY CHECK")
    report.append("")
    report.append("Formula used:")
    report.append("lambda_jr = lambda_0 * exp(gamma * U_r) / (I_jr + epsilon)^delta")
    report.append("")
    report.append(f"lambda_0 = {LAMBDA_0}")
    report.append(f"gamma = {GAMMA}")
    report.append(f"delta = {DELTA}")
    report.append(f"epsilon = {EPSILON}")
    report.append("")

    report.append("TABLE 1: Regime uncertainty scores")
    report.append(uncertainty_table.round(6).to_string(index=False))
    report.append("")

    report.append("TABLE 2: Feature evidence scores I_jr")
    report.append(evidence_matrix.to_string())
    report.append("")

    report.append("TABLE 3: Generated shrinkage matrix lambda_jr")
    report.append(lambda_matrix.to_string())
    report.append("")

    report.append("How to read lambda_jr:")
    report.append("Smaller lambda = weaker shrinkage = feature is more likely to survive.")
    report.append("Larger lambda = stronger shrinkage = feature is more likely to be removed.")
    report.append("")

    report.append("Lowest shrinkage feature in each regime:")
    for regime in lambda_matrix.columns:
        best_feature = lambda_matrix[regime].idxmin()
        best_lambda = lambda_matrix.loc[best_feature, regime]
        report.append(f"{regime}: {best_feature} with lambda = {best_lambda:.6f}")

    report.append("")
    report.append("Highest shrinkage feature in each regime:")
    for regime in lambda_matrix.columns:
        worst_feature = lambda_matrix[regime].idxmax()
        worst_lambda = lambda_matrix.loc[worst_feature, regime]
        report.append(f"{regime}: {worst_feature} with lambda = {worst_lambda:.6f}")

    return "\n".join(report)


def save_outputs(uncertainty_table, evidence_matrix, lambda_matrix):
    """Save all feasibility outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    uncertainty_file = OUTPUT_DIR / "table1_regime_uncertainty_scores.csv"
    evidence_file = OUTPUT_DIR / "table2_feature_evidence_scores.csv"
    lambda_file = OUTPUT_DIR / "table3_lambda_shrinkage_matrix.csv"
    interpretation_file = OUTPUT_DIR / "prior_feasibility_interpretation.txt"

    uncertainty_table.round(6).to_csv(uncertainty_file, index=False)
    evidence_matrix.to_csv(evidence_file)
    lambda_matrix.to_csv(lambda_file)

    interpretation = create_interpretation(
        uncertainty_table,
        evidence_matrix,
        lambda_matrix,
    )
    interpretation_file.write_text(interpretation, encoding="utf-8")

    return {
        "uncertainty": uncertainty_file,
        "evidence": evidence_file,
        "lambda": lambda_file,
        "interpretation": interpretation_file,
    }


def main():
    print("PRIOR FEASIBILITY CHECK")
    print("No Bayesian sampler is used here.")

    uncertainty_table = calculate_regime_uncertainty()
    _, evidence_matrix = calculate_feature_evidence()
    lambda_matrix = calculate_lambda_matrix(uncertainty_table, evidence_matrix)

    outputs = save_outputs(uncertainty_table, evidence_matrix, lambda_matrix)

    print("\nTABLE 1: Regime uncertainty scores")
    print(uncertainty_table.round(6).to_string(index=False))

    print("\nTABLE 2: Feature evidence scores I_jr")
    print(evidence_matrix.to_string())

    print("\nTABLE 3: Generated shrinkage matrix lambda_jr")
    print(lambda_matrix.to_string())

    print("\nSaved files:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
