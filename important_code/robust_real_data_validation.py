"""
Robust validation step 1: stronger real-data baselines and ablation study.

Why this script exists:

A reviewer may ask whether the proposed prior is better than obvious
alternatives, and whether both parts of the prior are actually needed.

This script compares:

1. Classical LASSO
2. Elastic Net
3. Adaptive LASSO
4. Regime-specific LASSO
5. Standard Bayesian LASSO
6. Regime-specific Bayesian LASSO
7. Proposed prior: uncertainty only
8. Proposed prior: evidence only
9. Proposed prior: uncertainty + evidence

The ablation logic is:

    Full prior:
        lambda_jr = exp(gamma * U_r) / (I_jr + epsilon)^delta

    Uncertainty only:
        lambda_jr = exp(gamma * U_r)

    Evidence only:
        lambda_jr = 1 / (I_jr + epsilon)^delta

This is still intentionally simple and explainable.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from bayesian_lasso_experiment import BayesianLassoGibbs
from simulation_study import WeightedBayesianLassoGibbs


DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "final_dataset_with_regimes.csv"
OUTPUT_DIR = DATA_DIR / "robust_real_data_validation"

TARGET_COLUMN = "Future_5_Day_Return"
MARKET_TICKER = "SPY"
RANDOM_STATE = 42

REGIME_ORDER = ["Recovery", "Bull", "Bear", "Crisis"]

GAMMA = 1.0
DELTA = 0.1
EPSILON = 0.001

MAX_GLOBAL_TRAIN_ROWS = 22000
MAX_REGIME_TRAIN_ROWS = 6000

MCMC_ITERATIONS = 700
BURN_IN = 300
THIN = 5

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


def load_data():
    """Load the final market dataset."""
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def add_research_features(df):
    """Create the same target and relative indicators used earlier."""
    result = df.copy()

    result["Future_Close_5"] = result.groupby("Ticker")["Close"].shift(-5)
    result[TARGET_COLUMN] = (result["Future_Close_5"] / result["Close"]) - 1

    result["Normalized_ATR"] = result["ATR_14"] / result["Close"]

    result["Close_to_SMA_10"] = (result["Close"] / result["SMA_10"]) - 1
    result["Close_to_SMA_20"] = (result["Close"] / result["SMA_20"]) - 1
    result["Close_to_SMA_50"] = (result["Close"] / result["SMA_50"]) - 1

    result["Close_to_EMA_10"] = (result["Close"] / result["EMA_10"]) - 1
    result["Close_to_EMA_20"] = (result["Close"] / result["EMA_20"]) - 1
    result["Close_to_EMA_50"] = (result["Close"] / result["EMA_50"]) - 1

    result["OBV_Change"] = result.groupby("Ticker")["OBV"].pct_change()
    result["OBV_Change"] = result["OBV_Change"].replace([np.inf, -np.inf], np.nan)

    return result


def prepare_model_data(df):
    """Keep complete rows for modelling."""
    columns = ["Date", "Ticker", "Regime", TARGET_COLUMN] + FEATURE_COLUMNS
    model_data = df[columns].copy()
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS)
    model_data = model_data.sort_values("Date").reset_index(drop=True)
    return model_data


def split_train_validation_test(model_data):
    """Use a chronological 60/20/20 split."""
    unique_dates = np.array(sorted(model_data["Date"].unique()))

    train_end = int(len(unique_dates) * 0.60)
    validation_end = int(len(unique_dates) * 0.80)

    train_date = unique_dates[train_end]
    validation_date = unique_dates[validation_end]

    train_data = model_data[model_data["Date"] < train_date].copy()
    validation_data = model_data[
        (model_data["Date"] >= train_date) & (model_data["Date"] < validation_date)
    ].copy()
    test_data = model_data[model_data["Date"] >= validation_date].copy()

    return train_data, validation_data, test_data, train_date, validation_date


def sample_rows(df, maximum_rows, seed):
    """Sample rows only when needed."""
    if len(df) <= maximum_rows:
        return df.copy()

    return df.sample(maximum_rows, random_state=seed).copy()


def scale_train_test(train_data, test_data):
    """Scale X and y using training data only."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train_data[FEATURE_COLUMNS])
    x_test = x_scaler.transform(test_data[FEATURE_COLUMNS])

    y_train = y_scaler.fit_transform(train_data[[TARGET_COLUMN]]).ravel()
    y_test = test_data[TARGET_COLUMN].to_numpy()

    return x_train, x_test, y_train, y_test, y_scaler


def invert_y_scale(predictions, y_scaler):
    """Return predictions to original return units."""
    return y_scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate model performance metrics."""
    return {
        "Model": model_name,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Direction_Accuracy": np.mean(np.sign(y_true) == np.sign(y_pred)),
        "Test_Rows": len(y_true),
    }


def selected_features_from_coef(coef):
    """Select non-trivial coefficients using a simple relative threshold."""
    max_abs = np.max(np.abs(coef))
    threshold = max(0.02, 0.15 * max_abs)
    return set(np.array(FEATURE_COLUMNS)[np.abs(coef) >= threshold])


def add_selected_rows(selected_rows, model_name, selected, regime="All"):
    """Append selected-feature rows."""
    for feature in sorted(selected):
        selected_rows.append(
            {
                "Model": model_name,
                "Regime": regime,
                "Feature": feature,
            }
        )


def train_lasso(train_data, test_data, selected_rows):
    """Train global LASSO."""
    train_sample = sample_rows(train_data, MAX_GLOBAL_TRAIN_ROWS, RANDOM_STATE)
    x_train, x_test, y_train, y_test, y_scaler = scale_train_test(train_sample, test_data)

    model = LassoCV(
        alphas=np.logspace(-5, -2, 35),
        cv=5,
        max_iter=20000,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = invert_y_scale(model.predict(x_test), y_scaler)
    selected = selected_features_from_coef(model.coef_)
    add_selected_rows(selected_rows, "Classical LASSO", selected)

    return calculate_metrics(y_test, predictions, "Classical LASSO")


def train_elastic_net(train_data, test_data, selected_rows):
    """Train global Elastic Net."""
    train_sample = sample_rows(train_data, MAX_GLOBAL_TRAIN_ROWS, RANDOM_STATE)
    x_train, x_test, y_train, y_test, y_scaler = scale_train_test(train_sample, test_data)

    model = ElasticNetCV(
        l1_ratio=[0.2, 0.5, 0.8, 0.95],
        alphas=np.logspace(-5, -2, 30),
        cv=5,
        max_iter=20000,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = invert_y_scale(model.predict(x_test), y_scaler)
    selected = selected_features_from_coef(model.coef_)
    add_selected_rows(selected_rows, "Elastic Net", selected)

    return calculate_metrics(y_test, predictions, "Elastic Net")


def train_adaptive_lasso(train_data, test_data, selected_rows):
    """Train a simple two-stage adaptive LASSO."""
    train_sample = sample_rows(train_data, MAX_GLOBAL_TRAIN_ROWS, RANDOM_STATE)
    x_train, x_test, y_train, y_test, y_scaler = scale_train_test(train_sample, test_data)

    first_stage = LassoCV(
        alphas=np.logspace(-5, -2, 35),
        cv=5,
        max_iter=20000,
        random_state=RANDOM_STATE,
    )
    first_stage.fit(x_train, y_train)

    weights = 1 / (np.abs(first_stage.coef_) + 0.01)
    weighted_x_train = x_train / weights
    weighted_x_test = x_test / weights

    second_stage = LassoCV(
        alphas=np.logspace(-5, -2, 35),
        cv=5,
        max_iter=20000,
        random_state=RANDOM_STATE + 1,
    )
    second_stage.fit(weighted_x_train, y_train)

    coef = second_stage.coef_ / weights
    predictions = invert_y_scale(weighted_x_test @ second_stage.coef_, y_scaler)
    selected = selected_features_from_coef(coef)
    add_selected_rows(selected_rows, "Adaptive LASSO", selected)

    return calculate_metrics(y_test, predictions, "Adaptive LASSO")


def train_standard_bayesian_lasso(train_data, test_data, selected_rows):
    """Train global Bayesian LASSO."""
    train_sample = sample_rows(train_data, MAX_GLOBAL_TRAIN_ROWS, RANDOM_STATE)
    x_train, x_test, y_train, y_test, y_scaler = scale_train_test(train_sample, test_data)

    model = BayesianLassoGibbs(
        lambda_value=1.0,
        n_iterations=MCMC_ITERATIONS,
        burn_in=BURN_IN,
        thin=THIN,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = invert_y_scale(model.predict(x_test), y_scaler)
    selected = selected_features_from_coef(model.coef_)
    add_selected_rows(selected_rows, "Standard Bayesian LASSO", selected)

    return calculate_metrics(y_test, predictions, "Standard Bayesian LASSO")


def train_regime_lasso(train_data, test_data, selected_rows):
    """Train separate classical LASSO models by regime."""
    predictions = []

    for regime in REGIME_ORDER:
        regime_train = train_data[train_data["Regime"] == regime].copy()
        regime_test = test_data[test_data["Regime"] == regime].copy()

        if regime_train.empty or regime_test.empty:
            continue

        train_sample = sample_rows(
            regime_train,
            MAX_REGIME_TRAIN_ROWS,
            RANDOM_STATE + REGIME_ORDER.index(regime),
        )
        x_train, x_test, y_train, y_test, y_scaler = scale_train_test(train_sample, regime_test)

        model = LassoCV(
            alphas=np.logspace(-5, -2, 30),
            cv=5,
            max_iter=20000,
            random_state=RANDOM_STATE + REGIME_ORDER.index(regime),
        )
        model.fit(x_train, y_train)

        regime_predictions = invert_y_scale(model.predict(x_test), y_scaler)
        frame = regime_test[[TARGET_COLUMN]].copy()
        frame["Prediction"] = regime_predictions
        predictions.append(frame)

        selected = selected_features_from_coef(model.coef_)
        add_selected_rows(selected_rows, "Regime-specific LASSO", selected, regime)

    all_predictions = pd.concat(predictions, ignore_index=True)
    return calculate_metrics(
        all_predictions[TARGET_COLUMN],
        all_predictions["Prediction"],
        "Regime-specific LASSO",
    )


def train_regime_bayesian_model(train_data, test_data, selected_rows, lambda_matrix, model_name):
    """Train one weighted Bayesian LASSO per regime."""
    predictions = []

    for regime in REGIME_ORDER:
        regime_train = train_data[train_data["Regime"] == regime].copy()
        regime_test = test_data[test_data["Regime"] == regime].copy()

        if regime_train.empty or regime_test.empty:
            continue

        train_sample = sample_rows(
            regime_train,
            MAX_REGIME_TRAIN_ROWS,
            RANDOM_STATE + REGIME_ORDER.index(regime),
        )
        x_train, x_test, y_train, y_test, y_scaler = scale_train_test(train_sample, regime_test)

        model = WeightedBayesianLassoGibbs(
            lambda_values=lambda_matrix[regime].loc[FEATURE_COLUMNS].to_numpy(),
            n_iterations=MCMC_ITERATIONS,
            burn_in=BURN_IN,
            thin=THIN,
            random_state=RANDOM_STATE + 100 + REGIME_ORDER.index(regime),
        )
        model.fit(x_train, y_train)

        regime_predictions = invert_y_scale(model.predict(x_test), y_scaler)
        frame = regime_test[[TARGET_COLUMN]].copy()
        frame["Prediction"] = regime_predictions
        predictions.append(frame)

        selected = selected_features_from_coef(model.coef_)
        add_selected_rows(selected_rows, model_name, selected, regime)

    all_predictions = pd.concat(predictions, ignore_index=True)
    return calculate_metrics(
        all_predictions[TARGET_COLUMN],
        all_predictions["Prediction"],
        model_name,
    )


def z_score(series):
    """Population z-score."""
    return (series - series.mean()) / series.std(ddof=0)


def calculate_regime_uncertainty(train_data):
    """Calculate U_r from training data only."""
    market = train_data[train_data["Ticker"] == MARKET_TICKER].copy()

    rows = []
    for regime in REGIME_ORDER:
        regime_rows = market[market["Regime"] == regime]
        volatility = regime_rows[TARGET_COLUMN].std()

        ordered = market.sort_values("Date")
        same_regime_days = ordered[
            (ordered["Regime"] == regime) & (ordered["Regime"].shift(-1) == regime)
        ]
        total_regime_days = ordered[ordered["Regime"] == regime]
        persistence = len(same_regime_days) / len(total_regime_days) if len(total_regime_days) else 0

        rows.append(
            {
                "Regime": regime,
                "Volatility": volatility,
                "Persistence": persistence,
            }
        )

    table = pd.DataFrame(rows)
    table["Instability"] = 1 - table["Persistence"]
    table["U_r"] = z_score(table["Volatility"]) + z_score(table["Instability"])
    return table


def calculate_feature_evidence(train_data):
    """Estimate feature evidence I_jr using absolute training correlations."""
    rows = []

    for regime in REGIME_ORDER:
        regime_rows = train_data[train_data["Regime"] == regime]

        for feature in FEATURE_COLUMNS:
            correlation = regime_rows[feature].corr(regime_rows[TARGET_COLUMN])
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


def calculate_lambda_matrix(uncertainty, evidence, mode):
    """Create lambda_jr for full prior or ablation variants."""
    u_values = uncertainty.set_index("Regime")["U_r"]
    lambda_matrix = evidence.copy().astype(float)

    for regime in REGIME_ORDER:
        if mode == "full":
            lambda_matrix[regime] = (
                np.exp(GAMMA * u_values.loc[regime])
                / ((evidence[regime] + EPSILON) ** DELTA)
            )
        elif mode == "uncertainty_only":
            lambda_matrix[regime] = np.exp(GAMMA * u_values.loc[regime])
        elif mode == "evidence_only":
            lambda_matrix[regime] = 1 / ((evidence[regime] + EPSILON) ** DELTA)
        elif mode == "constant":
            lambda_matrix[regime] = 1.0
        else:
            raise ValueError(f"Unknown lambda mode: {mode}")

    return lambda_matrix


def save_outputs(model_results, selected_features, uncertainty, evidence):
    """Save all robust validation outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_file = OUTPUT_DIR / "real_data_stronger_baselines.csv"
    selected_file = OUTPUT_DIR / "real_data_selected_features.csv"
    uncertainty_file = OUTPUT_DIR / "real_data_uncertainty_scores.csv"
    evidence_file = OUTPUT_DIR / "real_data_feature_evidence.csv"
    interpretation_file = OUTPUT_DIR / "real_data_validation_interpretation.txt"

    model_results.round(6).to_csv(model_file, index=False)
    selected_features.to_csv(selected_file, index=False)
    uncertainty.round(6).to_csv(uncertainty_file, index=False)
    evidence.round(6).to_csv(evidence_file)

    lines = []
    lines.append("ROBUST REAL-DATA VALIDATION")
    lines.append("")
    lines.append("Purpose:")
    lines.append("Compare stronger baselines and ablate the proposed prior.")
    lines.append("")
    lines.append("Model results:")
    lines.append(model_results.round(6).to_string(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "If the full prior beats the uncertainty-only and evidence-only variants, "
        "both parts of the proposed formula are useful."
    )
    lines.append(
        "If simpler baselines beat the proposed model on real data, the result "
        "should be explained using the weak-regime simulation finding."
    )

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")

    return {
        "models": model_file,
        "selected": selected_file,
        "uncertainty": uncertainty_file,
        "evidence": evidence_file,
        "interpretation": interpretation_file,
    }


def main():
    print("ROBUST REAL-DATA VALIDATION")
    print("Step 1: stronger baselines plus ablation study")

    raw_data = load_data()
    data_with_features = add_research_features(raw_data)
    model_data = prepare_model_data(data_with_features)

    train_data, validation_data, test_data, train_date, validation_date = split_train_validation_test(
        model_data
    )

    # Use train + validation for the final comparison after the split is fixed.
    final_train = pd.concat([train_data, validation_data], ignore_index=True)

    print(f"\nRows available: {len(model_data)}")
    print(f"Training rows: {len(final_train)}")
    print(f"Testing rows: {len(test_data)}")
    print(f"Initial train end date: {pd.Timestamp(train_date).date()}")
    print(f"Final test start date: {pd.Timestamp(validation_date).date()}")

    selected_rows = []
    result_rows = []

    print("\nTraining Classical LASSO")
    result_rows.append(train_lasso(final_train, test_data, selected_rows))

    print("Training Elastic Net")
    result_rows.append(train_elastic_net(final_train, test_data, selected_rows))

    print("Training Adaptive LASSO")
    result_rows.append(train_adaptive_lasso(final_train, test_data, selected_rows))

    print("Training Regime-specific LASSO")
    result_rows.append(train_regime_lasso(final_train, test_data, selected_rows))

    print("Training Standard Bayesian LASSO")
    result_rows.append(train_standard_bayesian_lasso(final_train, test_data, selected_rows))

    uncertainty = calculate_regime_uncertainty(final_train)
    evidence = calculate_feature_evidence(final_train)

    print("Training Regime-specific Bayesian LASSO")
    constant_lambda = calculate_lambda_matrix(uncertainty, evidence, "constant")
    result_rows.append(
        train_regime_bayesian_model(
            final_train,
            test_data,
            selected_rows,
            constant_lambda,
            "Regime-specific Bayesian LASSO",
        )
    )

    print("Training Proposed Prior: uncertainty only")
    uncertainty_lambda = calculate_lambda_matrix(uncertainty, evidence, "uncertainty_only")
    result_rows.append(
        train_regime_bayesian_model(
            final_train,
            test_data,
            selected_rows,
            uncertainty_lambda,
            "Proposed Prior - Uncertainty Only",
        )
    )

    print("Training Proposed Prior: evidence only")
    evidence_lambda = calculate_lambda_matrix(uncertainty, evidence, "evidence_only")
    result_rows.append(
        train_regime_bayesian_model(
            final_train,
            test_data,
            selected_rows,
            evidence_lambda,
            "Proposed Prior - Evidence Only",
        )
    )

    print("Training Proposed Prior: uncertainty + evidence")
    full_lambda = calculate_lambda_matrix(uncertainty, evidence, "full")
    result_rows.append(
        train_regime_bayesian_model(
            final_train,
            test_data,
            selected_rows,
            full_lambda,
            "Proposed Prior - Full",
        )
    )

    model_results = pd.DataFrame(result_rows)
    model_results = model_results.sort_values("RMSE").reset_index(drop=True)
    model_results["Rank"] = np.arange(1, len(model_results) + 1)

    selected_features = pd.DataFrame(selected_rows)

    outputs = save_outputs(model_results, selected_features, uncertainty, evidence)

    print("\nREAL-DATA MODEL COMPARISON")
    print(model_results.round(6).to_string(index=False))

    print("\nSaved files:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
