"""
Step 2 scientific verification: simulation study.

Goal:

    Check when regime-adaptive sparsity helps.

Scenarios:

1. Weak regime differences, 5000 samples per regime
2. Weak regime differences, 10000 samples per regime
3. Strong regime differences, 5000 samples per regime
4. Strong regime differences, 10000 samples per regime

Models compared:

1. Classical LASSO
2. Standard Bayesian LASSO
3. Regime-Uncertainty Weighted Bayesian LASSO

The simulation is intentionally simple. It is not trying to reproduce the stock
market exactly. It is testing whether the proposed prior can recover different
sparse feature patterns when those patterns really exist.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from bayesian_lasso_experiment import BayesianLassoGibbs


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "simulation_study"

RANDOM_STATE = 42

REGIME_ORDER = ["Recovery", "Bull", "Bear", "Crisis"]

FEATURE_COLUMNS = [
    "Daily_Return",
    "Log_Return",
    "Momentum_10",
    "ROC_10",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "Close_to_SMA_10",
    "Close_to_SMA_20",
    "Close_to_SMA_50",
    "Close_to_EMA_10",
    "Close_to_EMA_20",
    "Close_to_EMA_50",
    "ATR_14",
    "Normalized_ATR",
    "Volatility_20",
    "Bollinger_Width",
    "Bollinger_Position",
    "Volume_Ratio",
    "OBV_Change",
]

GAMMA = 1.0
DELTA = 1.0
EPSILON = 0.001
LAMBDA_0 = 1.0

MCMC_ITERATIONS = 500
BURN_IN = 200
THIN = 5

MAX_GLOBAL_TRAIN_ROWS = 10000
MAX_REGIME_TRAIN_ROWS = 3500


class WeightedBayesianLassoGibbs(BayesianLassoGibbs):
    """
    Bayesian LASSO with a different lambda value for each feature.

    This is used only for the simulation study. The full real-data model can be
    built after the simulation results are examined.
    """

    def __init__(
        self,
        lambda_values,
        n_iterations=500,
        burn_in=200,
        thin=5,
        random_state=42,
    ):
        super().__init__(
            lambda_value=1.0,
            n_iterations=n_iterations,
            burn_in=burn_in,
            thin=thin,
            random_state=random_state,
        )
        self.lambda_values = np.asarray(lambda_values)

    def _sample_tau_squared(self, rng, beta, sigma_squared, lambda_squared):
        """Use feature-specific lambda values in the local shrinkage update."""
        lambda_squared_vector = self.lambda_values ** 2

        safe_beta = np.maximum(np.abs(beta), 1e-8)
        mean = np.sqrt(lambda_squared_vector * sigma_squared / (safe_beta ** 2))
        mean = np.clip(mean, 1e-6, 1e6)

        inverse_tau_squared = rng.wald(mean=mean, scale=lambda_squared_vector)
        inverse_tau_squared = np.clip(inverse_tau_squared, 1e-8, 1e8)

        tau_squared = 1 / inverse_tau_squared
        tau_squared = np.clip(tau_squared, 1e-8, 1e8)
        return tau_squared


def make_feature_covariance(n_features, rho=0.30):
    """Create a simple correlated feature covariance matrix."""
    covariance = np.full((n_features, n_features), rho)
    np.fill_diagonal(covariance, 1.0)
    return covariance


def true_coefficients(case_name):
    """Create true sparse coefficients for weak or strong regime differences."""
    p = len(FEATURE_COLUMNS)
    coefficients = {regime: np.zeros(p) for regime in REGIME_ORDER}

    feature_index = {feature: i for i, feature in enumerate(FEATURE_COLUMNS)}

    if case_name == "Weak":
        common_features = {
            "Normalized_ATR": 0.45,
            "Volatility_20": -0.35,
            "RSI_14": 0.25,
            "Volume_Ratio": 0.20,
        }

        for regime in REGIME_ORDER:
            for feature, value in common_features.items():
                coefficients[regime][feature_index[feature]] = value

        coefficients["Recovery"][feature_index["Momentum_10"]] = 0.15
        coefficients["Bull"][feature_index["Close_to_EMA_20"]] = 0.15
        coefficients["Bear"][feature_index["ATR_14"]] = -0.15
        coefficients["Crisis"][feature_index["MACD"]] = -0.15

    else:
        active_sets = {
            "Recovery": {
                "RSI_14": 0.65,
                "Momentum_10": 0.55,
                "OBV_Change": 0.45,
            },
            "Bull": {
                "Volume_Ratio": 0.60,
                "Close_to_EMA_20": 0.50,
                "MACD": 0.45,
            },
            "Bear": {
                "ATR_14": -0.70,
                "Volatility_20": -0.60,
                "Bollinger_Width": -0.45,
            },
            "Crisis": {
                "Normalized_ATR": -0.80,
                "Daily_Return": -0.55,
                "MACD_Histogram": -0.45,
                "OBV_Change": 0.40,
            },
        }

        for regime, feature_values in active_sets.items():
            for feature, value in feature_values.items():
                coefficients[regime][feature_index[feature]] = value

    return coefficients


def regime_noise(case_name):
    """Set regime noise levels."""
    if case_name == "Weak":
        return {
            "Recovery": 1.00,
            "Bull": 1.05,
            "Bear": 1.10,
            "Crisis": 1.15,
        }

    return {
        "Recovery": 0.80,
        "Bull": 1.00,
        "Bear": 1.25,
        "Crisis": 1.60,
    }


def regime_persistence(case_name):
    """Set simple regime persistence values for the uncertainty score."""
    if case_name == "Weak":
        return {
            "Recovery": 0.96,
            "Bull": 0.95,
            "Bear": 0.94,
            "Crisis": 0.93,
        }

    return {
        "Recovery": 0.98,
        "Bull": 0.95,
        "Bear": 0.92,
        "Crisis": 0.90,
    }


def simulate_dataset(case_name, samples_per_regime, seed):
    """Generate synthetic regression data with known regime-specific sparsity."""
    rng = np.random.default_rng(seed)
    n_features = len(FEATURE_COLUMNS)
    covariance = make_feature_covariance(n_features)
    betas = true_coefficients(case_name)
    noise = regime_noise(case_name)

    frames = []

    for regime in REGIME_ORDER:
        x = rng.multivariate_normal(
            mean=np.zeros(n_features),
            cov=covariance,
            size=samples_per_regime,
        )
        error = rng.normal(0, noise[regime], size=samples_per_regime)
        y = x @ betas[regime] + error

        frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
        frame["Target"] = y
        frame["Regime"] = regime
        frame["Time_Index"] = np.arange(samples_per_regime)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    return data, betas, noise, regime_persistence(case_name)


def make_train_test_split(data):
    """Use the first 75 percent within each regime for training."""
    train_frames = []
    test_frames = []

    for regime, group in data.groupby("Regime"):
        group = group.sort_values("Time_Index")
        split_position = int(len(group) * 0.75)
        train_frames.append(group.iloc[:split_position])
        test_frames.append(group.iloc[split_position:])

    train_data = pd.concat(train_frames, ignore_index=True)
    test_data = pd.concat(test_frames, ignore_index=True)
    return train_data, test_data


def sample_rows(df, max_rows, seed):
    """Sample rows for faster Bayesian fitting."""
    if len(df) <= max_rows:
        return df.copy()

    return df.sample(max_rows, random_state=seed).copy()


def scale_train_test(train_data, test_data):
    """Scale X and y using training data only."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train_data[FEATURE_COLUMNS])
    x_test = x_scaler.transform(test_data[FEATURE_COLUMNS])

    y_train = y_scaler.fit_transform(train_data[["Target"]]).ravel()
    y_test = test_data["Target"].to_numpy()

    return x_train, x_test, y_train, y_test, y_scaler


def invert_y_scale(predictions, y_scaler):
    """Convert standardized predictions back to original scale."""
    return y_scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()


def metrics(y_true, y_pred):
    """Calculate prediction metrics."""
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Direction_Accuracy": np.mean(np.sign(y_true) == np.sign(y_pred)),
    }


def selected_from_coefficients(coefficients):
    """Select features using a simple relative coefficient threshold."""
    max_abs = np.max(np.abs(coefficients))
    threshold = max(0.02, 0.15 * max_abs)
    return set(np.array(FEATURE_COLUMNS)[np.abs(coefficients) >= threshold])


def true_support_for_regime(betas, regime):
    """Return true active features for one regime."""
    beta = betas[regime]
    return set(np.array(FEATURE_COLUMNS)[np.abs(beta) > 1e-8])


def true_union_support(betas):
    """Return features active in at least one regime."""
    support = set()
    for regime in REGIME_ORDER:
        support = support.union(true_support_for_regime(betas, regime))
    return support


def feature_recovery_scores(selected, true_support):
    """Calculate feature recovery precision, recall, and F1."""
    true_positive = len(selected.intersection(true_support))
    false_positive = len(selected - true_support)
    false_negative = len(true_support - selected)

    precision = true_positive / (true_positive + false_positive) if selected else 0.0
    recall = true_positive / (true_positive + false_negative) if true_support else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Selected_Count": len(selected),
        "True_Count": len(true_support),
    }


def train_lasso(train_data, test_data, betas, seed):
    """Train classical LASSO on all regimes together."""
    train_sample = sample_rows(train_data, MAX_GLOBAL_TRAIN_ROWS, seed)
    x_train, x_test, y_train, y_test, y_scaler = scale_train_test(train_sample, test_data)

    model = LassoCV(
        alphas=np.logspace(-4, 0, 35),
        cv=5,
        max_iter=20000,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    predictions = invert_y_scale(model.predict(x_test), y_scaler)
    selected = selected_from_coefficients(model.coef_)

    metric_row = metrics(y_test, predictions)
    recovery_row = feature_recovery_scores(selected, true_union_support(betas))

    return metric_row, recovery_row, selected


def train_standard_bayesian_lasso(train_data, test_data, betas, seed):
    """Train standard Bayesian LASSO on all regimes together."""
    train_sample = sample_rows(train_data, MAX_GLOBAL_TRAIN_ROWS, seed)
    x_train, x_test, y_train, y_test, y_scaler = scale_train_test(train_sample, test_data)

    model = BayesianLassoGibbs(
        lambda_value=1.0,
        n_iterations=MCMC_ITERATIONS,
        burn_in=BURN_IN,
        thin=THIN,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    predictions = invert_y_scale(model.predict(x_test), y_scaler)
    selected = selected_from_coefficients(model.coef_)

    metric_row = metrics(y_test, predictions)
    recovery_row = feature_recovery_scores(selected, true_union_support(betas))

    return metric_row, recovery_row, selected


def z_score(values):
    """Population z-score for a pandas Series."""
    return (values - values.mean()) / values.std(ddof=0)


def estimate_feature_evidence(train_data):
    """
    Estimate I_jr using absolute feature-target correlations inside each regime.

    This mimics the idea of regime-wise feature evidence without using the true
    simulated coefficients.
    """
    rows = []

    for regime, group in train_data.groupby("Regime"):
        for feature in FEATURE_COLUMNS:
            correlation = group[feature].corr(group["Target"])
            if pd.isna(correlation):
                correlation = 0.0

            rows.append(
                {
                    "Regime": regime,
                    "Feature": feature,
                    "Raw_Evidence": abs(correlation),
                }
            )

    evidence = pd.DataFrame(rows)
    min_value = evidence["Raw_Evidence"].min()
    max_value = evidence["Raw_Evidence"].max()
    evidence["I_jr"] = (evidence["Raw_Evidence"] - min_value) / (max_value - min_value)

    matrix = evidence.pivot(index="Feature", columns="Regime", values="I_jr")
    return matrix[REGIME_ORDER]


def calculate_uncertainty(noise, persistence):
    """Calculate U_r from simulated volatility and persistence."""
    table = pd.DataFrame(
        {
            "Regime": REGIME_ORDER,
            "Volatility": [noise[regime] for regime in REGIME_ORDER],
            "Persistence": [persistence[regime] for regime in REGIME_ORDER],
        }
    )
    table["Instability"] = 1 - table["Persistence"]
    table["U_r"] = z_score(table["Volatility"]) + z_score(table["Instability"])
    return table


def calculate_lambda_matrix(evidence, uncertainty):
    """Calculate lambda_jr for the proposed simulation model."""
    u_values = uncertainty.set_index("Regime")["U_r"]
    lambda_matrix = evidence.copy().astype(float)

    for regime in REGIME_ORDER:
        lambda_matrix[regime] = (
            LAMBDA_0
            * np.exp(GAMMA * u_values.loc[regime])
            / ((lambda_matrix[regime] + EPSILON) ** DELTA)
        )

    return lambda_matrix


def train_proposed_model(train_data, test_data, betas, noise, persistence, seed):
    """Train the proposed prior as one weighted Bayesian LASSO per regime."""
    evidence = estimate_feature_evidence(train_data)
    uncertainty = calculate_uncertainty(noise, persistence)
    lambda_matrix = calculate_lambda_matrix(evidence, uncertainty)

    prediction_frames = []
    selected_by_regime = {}

    for regime in REGIME_ORDER:
        regime_train = train_data[train_data["Regime"] == regime].copy()
        regime_test = test_data[test_data["Regime"] == regime].copy()

        train_sample = sample_rows(
            regime_train,
            MAX_REGIME_TRAIN_ROWS,
            seed + REGIME_ORDER.index(regime),
        )
        x_train, x_test, y_train, y_test, y_scaler = scale_train_test(
            train_sample,
            regime_test,
        )

        model = WeightedBayesianLassoGibbs(
            lambda_values=lambda_matrix[regime].loc[FEATURE_COLUMNS].to_numpy(),
            n_iterations=MCMC_ITERATIONS,
            burn_in=BURN_IN,
            thin=THIN,
            random_state=seed + 100 + REGIME_ORDER.index(regime),
        )
        model.fit(x_train, y_train)

        predictions = invert_y_scale(model.predict(x_test), y_scaler)
        frame = regime_test[["Regime", "Target"]].copy()
        frame["Prediction"] = predictions
        prediction_frames.append(frame)

        selected_by_regime[regime] = selected_from_coefficients(model.coef_)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metric_row = metrics(predictions["Target"], predictions["Prediction"])

    recovery_rows = []
    for regime in REGIME_ORDER:
        row = feature_recovery_scores(
            selected_by_regime[regime],
            true_support_for_regime(betas, regime),
        )
        row["Regime"] = regime
        recovery_rows.append(row)

    recovery_summary = pd.DataFrame(recovery_rows).mean(numeric_only=True).to_dict()

    selected_rows = []
    for regime, features in selected_by_regime.items():
        for feature in sorted(features):
            selected_rows.append(
                {
                    "Regime": regime,
                    "Feature": feature,
                }
            )

    return metric_row, recovery_summary, selected_rows


def add_labels(row, model_name, case_name, samples_per_regime):
    """Add scenario labels to a result row."""
    labelled = dict(row)
    labelled["Model"] = model_name
    labelled["Case"] = case_name
    labelled["Samples_Per_Regime"] = samples_per_regime
    return labelled


def run_one_scenario(case_name, samples_per_regime, seed):
    """Run all models for one simulation scenario."""
    print(f"\nScenario: {case_name}, samples per regime: {samples_per_regime}")

    data, betas, noise, persistence = simulate_dataset(case_name, samples_per_regime, seed)
    train_data, test_data = make_train_test_split(data)

    metric_rows = []
    recovery_rows = []
    selected_rows = []

    print("Training Classical LASSO")
    metrics_lasso, recovery_lasso, selected_lasso = train_lasso(
        train_data,
        test_data,
        betas,
        seed,
    )
    metric_rows.append(add_labels(metrics_lasso, "Classical LASSO", case_name, samples_per_regime))
    recovery_rows.append(add_labels(recovery_lasso, "Classical LASSO", case_name, samples_per_regime))
    for feature in sorted(selected_lasso):
        selected_rows.append(
            {
                "Case": case_name,
                "Samples_Per_Regime": samples_per_regime,
                "Model": "Classical LASSO",
                "Regime": "All",
                "Feature": feature,
            }
        )

    print("Training Standard Bayesian LASSO")
    metrics_bayes, recovery_bayes, selected_bayes = train_standard_bayesian_lasso(
        train_data,
        test_data,
        betas,
        seed + 1,
    )
    metric_rows.append(add_labels(metrics_bayes, "Standard Bayesian LASSO", case_name, samples_per_regime))
    recovery_rows.append(add_labels(recovery_bayes, "Standard Bayesian LASSO", case_name, samples_per_regime))
    for feature in sorted(selected_bayes):
        selected_rows.append(
            {
                "Case": case_name,
                "Samples_Per_Regime": samples_per_regime,
                "Model": "Standard Bayesian LASSO",
                "Regime": "All",
                "Feature": feature,
            }
        )

    print("Training Proposed Prior")
    metrics_proposed, recovery_proposed, proposed_selected = train_proposed_model(
        train_data,
        test_data,
        betas,
        noise,
        persistence,
        seed + 2,
    )
    metric_rows.append(add_labels(metrics_proposed, "Proposed Prior", case_name, samples_per_regime))
    recovery_rows.append(add_labels(recovery_proposed, "Proposed Prior", case_name, samples_per_regime))
    for row in proposed_selected:
        row["Case"] = case_name
        row["Samples_Per_Regime"] = samples_per_regime
        row["Model"] = "Proposed Prior"
        selected_rows.append(row)

    return metric_rows, recovery_rows, selected_rows


def write_interpretation(model_comparison, feature_recovery):
    """Write a concise simulation interpretation."""
    interpretation_file = OUTPUT_DIR / "simulation_interpretation.txt"

    lines = []
    lines.append("SIMULATION STUDY INTERPRETATION")
    lines.append("")
    lines.append("Question:")
    lines.append("When does regime-adaptive sparsity help?")
    lines.append("")
    lines.append("Prediction metrics:")
    lines.append(model_comparison.round(6).to_string(index=False))
    lines.append("")
    lines.append("Feature recovery metrics:")
    lines.append(feature_recovery.round(6).to_string(index=False))
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("If the proposed prior improves F1 in the Strong case, it supports the")
    lines.append("claim that adaptive sparsity helps when true feature relevance differs")
    lines.append("strongly across regimes.")

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")
    return interpretation_file


def save_outputs(model_comparison, feature_recovery, selected_features):
    """Save simulation outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_file = OUTPUT_DIR / "simulation_model_comparison.csv"
    recovery_file = OUTPUT_DIR / "simulation_feature_recovery.csv"
    selected_file = OUTPUT_DIR / "simulation_selected_features.csv"

    model_comparison.round(6).to_csv(model_file, index=False)
    feature_recovery.round(6).to_csv(recovery_file, index=False)
    selected_features.to_csv(selected_file, index=False)

    interpretation_file = write_interpretation(model_comparison, feature_recovery)

    return {
        "model_comparison": model_file,
        "feature_recovery": recovery_file,
        "selected_features": selected_file,
        "interpretation": interpretation_file,
    }


def main():
    print("SIMULATION STUDY")
    print("Testing weak and strong regime differences.")

    all_metric_rows = []
    all_recovery_rows = []
    all_selected_rows = []

    scenarios = [
        ("Weak", 5000),
        ("Weak", 10000),
        ("Strong", 5000),
        ("Strong", 10000),
    ]

    for scenario_number, (case_name, samples_per_regime) in enumerate(scenarios):
        seed = RANDOM_STATE + scenario_number * 10
        metric_rows, recovery_rows, selected_rows = run_one_scenario(
            case_name,
            samples_per_regime,
            seed,
        )
        all_metric_rows.extend(metric_rows)
        all_recovery_rows.extend(recovery_rows)
        all_selected_rows.extend(selected_rows)

    model_comparison = pd.DataFrame(all_metric_rows)
    feature_recovery = pd.DataFrame(all_recovery_rows)
    selected_features = pd.DataFrame(all_selected_rows)

    output_files = save_outputs(model_comparison, feature_recovery, selected_features)

    print("\nSIMULATION MODEL COMPARISON")
    print(model_comparison.round(6).to_string(index=False))

    print("\nSIMULATION FEATURE RECOVERY")
    print(feature_recovery.round(6).to_string(index=False))

    print("\nSaved files:")
    for path in output_files.values():
        print(path)


if __name__ == "__main__":
    main()
