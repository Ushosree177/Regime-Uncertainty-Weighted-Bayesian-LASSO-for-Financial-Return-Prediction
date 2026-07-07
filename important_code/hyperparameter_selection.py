"""
Proposition 1A: automatic hyperparameter selection for the proposed prior.

Sensitivity analysis asks:

    Is the prior stable when gamma and delta change?

Hyperparameter selection asks:

    Which gamma and delta should we use?

This script uses a simple grid search:

    gamma in {0.1, 0.5, 1, 2, 5}
    delta in {0.1, 0.5, 1, 2, 5}

For each pair, it trains the proposed Regime-Uncertainty Weighted Bayesian
LASSO on synthetic training data and evaluates it on validation data.

The best pair is chosen by lowest validation RMSE.

This script is intentionally separate from the full simulation study because the
two questions are different:

    Sensitivity analysis = stability
    Hyperparameter selection = best values
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from simulation_study import (
    FEATURE_COLUMNS,
    REGIME_ORDER,
    WeightedBayesianLassoGibbs,
    calculate_uncertainty,
    estimate_feature_evidence,
    invert_y_scale,
    sample_rows,
    simulate_dataset,
)


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "hyperparameter_selection"

RANDOM_STATE = 42

GAMMA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
DELTA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]

LAMBDA_0 = 1.0
EPSILON = 0.001

# A smaller sampler is enough for tuning. After choosing gamma and delta, the
# final experiment can use more iterations.
MCMC_ITERATIONS = 300
BURN_IN = 120
THIN = 4

MAX_REGIME_TRAIN_ROWS = 3000

# Tuning is demonstrated on the most important simulation setting: strong regime
# differences. This is where adaptive sparsity should matter most.
CASE_NAME = "Strong"
SAMPLES_PER_REGIME = 5000


def split_train_validation_test(data):
    """Create 60 percent train, 20 percent validation, 20 percent test per regime."""
    train_frames = []
    validation_frames = []
    test_frames = []

    for regime, group in data.groupby("Regime"):
        group = group.sort_values("Time_Index")

        train_end = int(len(group) * 0.60)
        validation_end = int(len(group) * 0.80)

        train_frames.append(group.iloc[:train_end])
        validation_frames.append(group.iloc[train_end:validation_end])
        test_frames.append(group.iloc[validation_end:])

    train_data = pd.concat(train_frames, ignore_index=True)
    validation_data = pd.concat(validation_frames, ignore_index=True)
    test_data = pd.concat(test_frames, ignore_index=True)

    return train_data, validation_data, test_data


def calculate_lambda_matrix(evidence, uncertainty, gamma, delta):
    """Calculate lambda_jr for one gamma-delta pair."""
    u_values = uncertainty.set_index("Regime")["U_r"]
    lambda_matrix = evidence.copy().astype(float)

    for regime in REGIME_ORDER:
        lambda_matrix[regime] = (
            LAMBDA_0
            * np.exp(gamma * u_values.loc[regime])
            / ((lambda_matrix[regime] + EPSILON) ** delta)
        )

    return lambda_matrix


def scale_regime_data(train_data, evaluation_data):
    """Scale X and y using one regime's training data."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train_data[FEATURE_COLUMNS])
    x_eval = x_scaler.transform(evaluation_data[FEATURE_COLUMNS])

    y_train = y_scaler.fit_transform(train_data[["Target"]]).ravel()
    y_eval = evaluation_data["Target"].to_numpy()

    return x_train, x_eval, y_train, y_eval, y_scaler


def calculate_metrics(y_true, y_pred):
    """Calculate prediction metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Direction_Accuracy": np.mean(np.sign(y_true) == np.sign(y_pred)),
    }


def evaluate_gamma_delta(train_data, evaluation_data, lambda_matrix, seed):
    """Train the proposed prior and evaluate one gamma-delta setting."""
    prediction_frames = []

    for regime in REGIME_ORDER:
        regime_train = train_data[train_data["Regime"] == regime].copy()
        regime_eval = evaluation_data[evaluation_data["Regime"] == regime].copy()

        train_sample = sample_rows(
            regime_train,
            MAX_REGIME_TRAIN_ROWS,
            seed + REGIME_ORDER.index(regime),
        )

        x_train, x_eval, y_train, y_eval, y_scaler = scale_regime_data(
            train_sample,
            regime_eval,
        )

        model = WeightedBayesianLassoGibbs(
            lambda_values=lambda_matrix[regime].loc[FEATURE_COLUMNS].to_numpy(),
            n_iterations=MCMC_ITERATIONS,
            burn_in=BURN_IN,
            thin=THIN,
            random_state=seed + 100 + REGIME_ORDER.index(regime),
        )
        model.fit(x_train, y_train)

        predictions = invert_y_scale(model.predict(x_eval), y_scaler)

        frame = regime_eval[["Regime", "Target"]].copy()
        frame["Prediction"] = predictions
        prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    return calculate_metrics(predictions["Target"], predictions["Prediction"])


def run_grid_search(train_data, validation_data, noise, persistence):
    """Run grid search over gamma and delta."""
    evidence = estimate_feature_evidence(train_data)
    uncertainty = calculate_uncertainty(noise, persistence)

    rows = []

    for gamma in GAMMA_VALUES:
        for delta in DELTA_VALUES:
            print(f"Testing gamma={gamma}, delta={delta}")

            lambda_matrix = calculate_lambda_matrix(
                evidence,
                uncertainty,
                gamma,
                delta,
            )

            metric_row = evaluate_gamma_delta(
                train_data,
                validation_data,
                lambda_matrix,
                seed=RANDOM_STATE + int(gamma * 10) + int(delta * 100),
            )

            metric_row["Gamma"] = gamma
            metric_row["Delta"] = delta
            rows.append(metric_row)

    results = pd.DataFrame(rows)
    results = results.sort_values(["RMSE", "MAE"], ascending=[True, True]).reset_index(drop=True)
    results["Rank"] = np.arange(1, len(results) + 1)

    return results, evidence, uncertainty


def evaluate_best_on_test(train_data, validation_data, test_data, evidence, uncertainty, best_row):
    """Retrain using train plus validation, then evaluate the best pair on test data."""
    combined_train = pd.concat([train_data, validation_data], ignore_index=True)

    # Re-estimate evidence using all non-test data.
    final_evidence = estimate_feature_evidence(combined_train)

    lambda_matrix = calculate_lambda_matrix(
        final_evidence,
        uncertainty,
        gamma=float(best_row["Gamma"]),
        delta=float(best_row["Delta"]),
    )

    test_metrics = evaluate_gamma_delta(
        combined_train,
        test_data,
        lambda_matrix,
        seed=RANDOM_STATE + 999,
    )

    test_metrics["Gamma"] = float(best_row["Gamma"])
    test_metrics["Delta"] = float(best_row["Delta"])
    test_metrics["Selected_By"] = "Lowest validation RMSE"

    return pd.DataFrame([test_metrics])


def plot_rmse_heatmap(results):
    """Save a validation RMSE heatmap."""
    heatmap_file = OUTPUT_DIR / "validation_rmse_heatmap.png"

    pivot = results.pivot(index="Delta", columns="Gamma", values="RMSE")
    pivot = pivot.loc[DELTA_VALUES, GAMMA_VALUES]

    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(pivot.values, cmap="viridis_r", aspect="auto")

    ax.set_title("Validation RMSE by Gamma and Delta")
    ax.set_xlabel("Gamma")
    ax.set_ylabel("Delta")
    ax.set_xticks(np.arange(len(GAMMA_VALUES)))
    ax.set_xticklabels(GAMMA_VALUES)
    ax.set_yticks(np.arange(len(DELTA_VALUES)))
    ax.set_yticklabels(DELTA_VALUES)

    for i in range(len(DELTA_VALUES)):
        for j in range(len(GAMMA_VALUES)):
            ax.text(
                j,
                i,
                f"{pivot.values[i, j]:.3f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Validation RMSE")

    plt.tight_layout()
    plt.savefig(heatmap_file, dpi=200)
    plt.close()

    return heatmap_file


def write_interpretation(results, test_result):
    """Write a short explanation for the tuning result."""
    interpretation_file = OUTPUT_DIR / "hyperparameter_selection_interpretation.txt"
    best = results.iloc[0]

    lines = []
    lines.append("HYPERPARAMETER SELECTION INTERPRETATION")
    lines.append("")
    lines.append("Question:")
    lines.append("Can gamma and delta be selected automatically?")
    lines.append("")
    lines.append(f"Case used: {CASE_NAME}")
    lines.append(f"Samples per regime: {SAMPLES_PER_REGIME}")
    lines.append("")
    lines.append("Grid searched:")
    lines.append(f"Gamma values: {GAMMA_VALUES}")
    lines.append(f"Delta values: {DELTA_VALUES}")
    lines.append("")
    lines.append("Best validation setting:")
    lines.append(
        f"gamma = {best['Gamma']}, delta = {best['Delta']}, "
        f"validation RMSE = {best['RMSE']:.6f}"
    )
    lines.append("")
    lines.append("Top 10 validation settings:")
    lines.append(results.head(10).round(6).to_string(index=False))
    lines.append("")
    lines.append("Test result using best validation setting:")
    lines.append(test_result.round(6).to_string(index=False))
    lines.append("")
    lines.append("Research meaning:")
    lines.append(
        "Gamma and delta do not need to be chosen manually. They can be selected "
        "by validation RMSE using a standard grid-search procedure."
    )

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")
    return interpretation_file


def save_outputs(results, test_result):
    """Save all hyperparameter-selection outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    grid_file = OUTPUT_DIR / "gamma_delta_grid_search_results.csv"
    best_file = OUTPUT_DIR / "best_gamma_delta_test_result.csv"

    results.round(6).to_csv(grid_file, index=False)
    test_result.round(6).to_csv(best_file, index=False)

    heatmap_file = plot_rmse_heatmap(results)
    interpretation_file = write_interpretation(results, test_result)

    return {
        "grid": grid_file,
        "test": best_file,
        "heatmap": heatmap_file,
        "interpretation": interpretation_file,
    }


def main():
    print("HYPERPARAMETER SELECTION")
    print("Grid search for gamma and delta.")

    data, _, noise, persistence = simulate_dataset(
        CASE_NAME,
        SAMPLES_PER_REGIME,
        seed=RANDOM_STATE,
    )

    train_data, validation_data, test_data = split_train_validation_test(data)

    print(f"\nCase: {CASE_NAME}")
    print(f"Samples per regime: {SAMPLES_PER_REGIME}")
    print(f"Training rows: {len(train_data)}")
    print(f"Validation rows: {len(validation_data)}")
    print(f"Testing rows: {len(test_data)}")

    results, evidence, uncertainty = run_grid_search(
        train_data,
        validation_data,
        noise,
        persistence,
    )

    best_row = results.iloc[0]
    test_result = evaluate_best_on_test(
        train_data,
        validation_data,
        test_data,
        evidence,
        uncertainty,
        best_row,
    )

    output_files = save_outputs(results, test_result)

    print("\nGRID SEARCH RESULTS")
    print(results.round(6).to_string(index=False))

    print("\nBEST TEST RESULT")
    print(test_result.round(6).to_string(index=False))

    print("\nSaved files:")
    for path in output_files.values():
        print(path)


if __name__ == "__main__":
    main()
