"""
Stage 5: Standard Bayesian LASSO and Regime-Adaptive Bayesian LASSO.

This script is the next step after regime-wise feature importance.

Research question:

    If feature relevance changes across regimes, does a regime-adaptive
    Bayesian LASSO perform better than one global Bayesian LASSO?

The script trains three models:

1. Classical LASSO baseline
2. Standard Bayesian LASSO using all training observations
3. Regime-Adaptive Bayesian LASSO with one Bayesian LASSO per regime

Target:

    5-day future return

Features:

    The same 21 technical indicators used in the regime feature-importance study.

Outputs:

    data/bayesian_lasso/model_comparison.csv
    data/bayesian_lasso/standard_bayesian_lasso_coefficients.csv
    data/bayesian_lasso/regime_adaptive_coefficients_long.csv
    data/bayesian_lasso/selected_features_by_regime.csv
    data/bayesian_lasso/interpretation.txt


"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "final_dataset_with_regimes.csv"
OUTPUT_DIR = DATA_DIR / "bayesian_lasso"

TARGET_COLUMN = "Future_5_Day_Return"
RANDOM_STATE = 42

REGIME_ORDER = ["Recovery", "Bull", "Bear", "Crisis"]

# These caps keep the Gibbs sampler fast enough for a laptop.
# Increase them later if you want a heavier final experiment.
MAX_GLOBAL_TRAIN_ROWS = 25000
MAX_REGIME_TRAIN_ROWS = 7000

# Gibbs sampler settings. These are moderate settings for a first research run.
MCMC_ITERATIONS = 1400
BURN_IN = 600
THIN = 5

FEATURE_COLUMNS = [
    # Return and momentum
    "Daily_Return",
    "Log_Return",
    "Momentum_10",
    "ROC_10",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",

    # Trend features converted to relative values later
    "Close_to_SMA_10",
    "Close_to_SMA_20",
    "Close_to_SMA_50",
    "Close_to_EMA_10",
    "Close_to_EMA_20",
    "Close_to_EMA_50",

    # Volatility
    "ATR_14",
    "Normalized_ATR",
    "Volatility_20",
    "Bollinger_Width",
    "Bollinger_Position",

    # Volume
    "Volume_Ratio",
    "OBV_Change",
]


class BayesianLassoGibbs:
    """
    Simple Bayesian LASSO using Gibbs sampling.

    The Laplace prior is written as a normal-exponential mixture:

        beta_j | tau_j^2, sigma^2 ~ Normal(0, sigma^2 * tau_j^2)
        tau_j^2 ~ Exponential(lambda^2 / 2)

    This gives LASSO-like shrinkage, but the output is Bayesian because we keep
    posterior samples of the coefficients.
    """

    def __init__(
        self,
        lambda_value=1.0,
        n_iterations=1400,
        burn_in=600,
        thin=5,
        random_state=42,
    ):
        self.lambda_value = lambda_value
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thin = thin
        self.random_state = random_state

        self.beta_samples_ = None
        self.sigma_squared_samples_ = None
        self.coef_ = None
        self.lower_95_ = None
        self.upper_95_ = None
        self.diagnostics_ = None

    def fit(self, x, y):
        """Fit the Bayesian LASSO model on already-standardized data."""
        rng = np.random.default_rng(self.random_state)

        x = np.asarray(x)
        y = np.asarray(y)

        n_rows, n_features = x.shape
        xtx = x.T @ x
        xty = x.T @ y

        beta = np.zeros(n_features)
        tau_squared = np.ones(n_features)
        sigma_squared = 1.0
        lambda_squared = self.lambda_value ** 2

        saved_samples = []
        saved_sigma_squared = []

        for iteration in range(self.n_iterations):
            beta = self._sample_beta(
                rng,
                xtx,
                xty,
                tau_squared,
                sigma_squared,
            )
            beta = np.nan_to_num(beta, nan=0.0, posinf=1e6, neginf=-1e6)

            sigma_squared = self._sample_sigma_squared(
                rng,
                x,
                y,
                beta,
                tau_squared,
                n_rows,
                n_features,
            )
            if not np.isfinite(sigma_squared) or sigma_squared <= 0:
                sigma_squared = 1.0

            tau_squared = self._sample_tau_squared(
                rng,
                beta,
                sigma_squared,
                lambda_squared,
            )
            tau_squared = np.nan_to_num(tau_squared, nan=1.0, posinf=1e8, neginf=1e-8)

            keep_sample = iteration >= self.burn_in and (iteration - self.burn_in) % self.thin == 0
            if keep_sample:
                saved_samples.append(beta.copy())
                saved_sigma_squared.append(sigma_squared)

        self.beta_samples_ = np.array(saved_samples)
        self.sigma_squared_samples_ = np.array(saved_sigma_squared)
        self.coef_ = self.beta_samples_.mean(axis=0)
        self.lower_95_ = np.percentile(self.beta_samples_, 2.5, axis=0)
        self.upper_95_ = np.percentile(self.beta_samples_, 97.5, axis=0)
        self.diagnostics_ = self._summarize_mcmc_diagnostics()

        return self

    def _summarize_mcmc_diagnostics(self):
        """Summarize single-chain MCMC stability diagnostics for saved draws."""
        if self.beta_samples_ is None or len(self.beta_samples_) == 0:
            return {
                "Saved_Samples": 0,
                "Min_ESS": np.nan,
                "Median_ESS": np.nan,
                "Max_MCSE": np.nan,
                "Mean_Abs_Lag1_Autocorr": np.nan,
                "Sigma_Squared_Mean": np.nan,
                "Sigma_Squared_SD": np.nan,
                "Sigma_Squared_ESS": np.nan,
            }

        ess_values = []
        mcse_values = []
        lag1_values = []

        for column in range(self.beta_samples_.shape[1]):
            draws = self.beta_samples_[:, column]
            ess = self._effective_sample_size(draws)
            sd = np.std(draws, ddof=1) if len(draws) > 1 else 0.0
            mcse = sd / np.sqrt(max(ess, 1.0))
            lag1 = self._autocorrelation(draws, lag=1)

            ess_values.append(ess)
            mcse_values.append(mcse)
            lag1_values.append(abs(lag1) if np.isfinite(lag1) else np.nan)

        sigma_draws = self.sigma_squared_samples_
        sigma_sd = np.std(sigma_draws, ddof=1) if len(sigma_draws) > 1 else 0.0

        return {
            "Saved_Samples": int(len(self.beta_samples_)),
            "Min_ESS": float(np.nanmin(ess_values)),
            "Median_ESS": float(np.nanmedian(ess_values)),
            "Max_MCSE": float(np.nanmax(mcse_values)),
            "Mean_Abs_Lag1_Autocorr": float(np.nanmean(lag1_values)),
            "Sigma_Squared_Mean": float(np.mean(sigma_draws)),
            "Sigma_Squared_SD": float(sigma_sd),
            "Sigma_Squared_ESS": float(self._effective_sample_size(sigma_draws)),
        }

    @staticmethod
    def _autocorrelation(draws, lag):
        """Estimate autocorrelation at a given lag."""
        draws = np.asarray(draws, dtype=float)
        if len(draws) <= lag:
            return np.nan

        centered = draws - np.mean(draws)
        denominator = np.dot(centered, centered)
        if denominator <= 1e-12:
            return 0.0

        numerator = np.dot(centered[:-lag], centered[lag:])
        return numerator / denominator

    @classmethod
    def _effective_sample_size(cls, draws, max_lag=40):
        """
        Estimate effective sample size from positive autocorrelation pairs.

        This is a lightweight single-chain diagnostic. It is not a replacement
        for multi-chain R-hat, but it gives a reproducible stability summary.
        """
        draws = np.asarray(draws, dtype=float)
        n_draws = len(draws)
        if n_draws <= 2:
            return float(n_draws)

        autocorr_sum = 0.0
        upper_lag = min(max_lag, n_draws - 1)

        for lag in range(1, upper_lag + 1):
            rho = cls._autocorrelation(draws, lag)
            if not np.isfinite(rho) or rho <= 0:
                break
            autocorr_sum += rho

        ess = n_draws / (1.0 + 2.0 * autocorr_sum)
        return float(np.clip(ess, 1.0, n_draws))

    def predict(self, x):
        """Predict with posterior mean coefficients."""
        predictions = np.asarray(x) @ self.coef_
        return np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)

    def _sample_beta(self, rng, xtx, xty, tau_squared, sigma_squared):
        """Sample regression coefficients from a multivariate normal."""
        prior_precision = np.diag(1 / tau_squared)
        precision = xtx + prior_precision

        mean = np.linalg.solve(precision, xty)
        covariance = sigma_squared * np.linalg.inv(precision)
        covariance = (covariance + covariance.T) / 2
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=1e6, neginf=-1e6)

        for jitter in [1e-10, 1e-8, 1e-6, 1e-4]:
            try:
                stable_covariance = covariance + np.eye(len(mean)) * jitter
                return rng.multivariate_normal(mean, stable_covariance, check_valid="ignore")
            except np.linalg.LinAlgError:
                continue

        diagonal = np.clip(np.diag(covariance), 1e-8, 1e6)
        return mean + rng.normal(0, np.sqrt(diagonal))

    def _sample_sigma_squared(self, rng, x, y, beta, tau_squared, n_rows, n_features):
        """Sample residual variance from an inverse-gamma distribution."""
        residual = y - x @ beta
        residual_sum = residual @ residual
        prior_sum = np.sum((beta ** 2) / tau_squared)

        shape = 0.5 * (n_rows + n_features)
        scale = 0.5 * (residual_sum + prior_sum) + 1e-8

        gamma_draw = rng.gamma(shape=shape, scale=1 / scale)
        return 1 / gamma_draw

    def _sample_tau_squared(self, rng, beta, sigma_squared, lambda_squared):
        """
        Sample local shrinkage values.

        The full conditional for 1 / tau_j^2 is inverse Gaussian. NumPy calls
        this distribution Wald.
        """
        safe_beta = np.maximum(np.abs(beta), 1e-8)
        mean = np.sqrt(lambda_squared * sigma_squared / (safe_beta ** 2))
        mean = np.clip(mean, 1e-6, 1e6)

        inverse_tau_squared = rng.wald(mean=mean, scale=lambda_squared)
        inverse_tau_squared = np.clip(inverse_tau_squared, 1e-8, 1e8)

        tau_squared = 1 / inverse_tau_squared
        tau_squared = np.clip(tau_squared, 1e-8, 1e8)
        return tau_squared


def load_data():
    """Load final HMM dataset."""
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def add_research_features(df):
    """Create the same target and safer cross-stock features used in Stage 4."""
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
    """Keep complete rows for the modelling experiment."""
    model_columns = ["Date", "Ticker", "Regime", TARGET_COLUMN] + FEATURE_COLUMNS

    model_data = df[model_columns].copy()
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS)
    model_data = model_data.sort_values("Date").reset_index(drop=True)

    return model_data


def make_time_split(model_data):
    """
    Split by date instead of random rows.

    This is more honest for finance because the test set happens after the
    training period.
    """
    unique_dates = np.array(sorted(model_data["Date"].unique()))
    split_position = int(len(unique_dates) * 0.75)
    split_date = unique_dates[split_position]

    train_data = model_data[model_data["Date"] < split_date].copy()
    test_data = model_data[model_data["Date"] >= split_date].copy()

    return train_data, test_data, split_date


def sample_rows(df, maximum_rows):
    """Sample rows only when the dataset is larger than the chosen cap."""
    if len(df) <= maximum_rows:
        return df.copy()

    return df.sample(maximum_rows, random_state=RANDOM_STATE).copy()


def scale_train_and_test(train_data, test_data):
    """Standardize X and y using training data only."""
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train_data[FEATURE_COLUMNS])
    x_test = x_scaler.transform(test_data[FEATURE_COLUMNS])

    y_train = y_scaler.fit_transform(train_data[[TARGET_COLUMN]]).ravel()
    y_test = test_data[TARGET_COLUMN].to_numpy()

    return x_train, x_test, y_train, y_test, x_scaler, y_scaler


def invert_y_scale(y_scaled, y_scaler):
    """Convert standardized predictions back to return units."""
    return y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def calculate_metrics(y_true, y_pred, model_name):
    """Calculate prediction accuracy metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Direction_Accuracy": direction_accuracy,
        "Test_Rows": len(y_true),
    }


def make_coefficient_table(model_name, features, coefficients, lower_95=None, upper_95=None):
    """Create a readable coefficient table."""
    table = pd.DataFrame(
        {
            "Model": model_name,
            "Feature": features,
            "Posterior_Mean_Coefficient": coefficients,
        }
    )

    if lower_95 is not None and upper_95 is not None:
        table["Lower_95"] = lower_95
        table["Upper_95"] = upper_95

    threshold = choose_selection_threshold(coefficients)
    table["Selected"] = np.abs(table["Posterior_Mean_Coefficient"]) >= threshold
    table["Abs_Coefficient"] = table["Posterior_Mean_Coefficient"].abs()
    table = table.sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    table["Rank"] = np.arange(1, len(table) + 1)

    return table


def choose_selection_threshold(coefficients):
    """
    Pick a simple coefficient threshold for feature selection.

    Coefficients are on standardized X and standardized y. A feature is selected
    if its posterior mean is at least 15 percent of the largest coefficient, with
    a small minimum cutoff to avoid selecting numerical noise.
    """
    max_abs = np.max(np.abs(coefficients))
    return max(0.02, 0.15 * max_abs)


def train_classical_lasso(train_data, test_data):
    """Train a standard LASSO baseline."""
    train_sample = sample_rows(train_data, MAX_GLOBAL_TRAIN_ROWS)
    x_train, x_test, y_train, y_test, _, y_scaler = scale_train_and_test(train_sample, test_data)

    model = LassoCV(
        alphas=np.logspace(-5, -2, 30),
        cv=5,
        max_iter=20000,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = invert_y_scale(model.predict(x_test), y_scaler)
    metrics = calculate_metrics(y_test, predictions, "Classical LASSO")

    coef_table = pd.DataFrame(
        {
            "Model": "Classical LASSO",
            "Feature": FEATURE_COLUMNS,
            "Coefficient": model.coef_,
        }
    )
    coef_table["Selected"] = coef_table["Coefficient"].abs() > 1e-8
    coef_table["Abs_Coefficient"] = coef_table["Coefficient"].abs()
    coef_table = coef_table.sort_values("Abs_Coefficient", ascending=False).reset_index(drop=True)
    coef_table["Rank"] = np.arange(1, len(coef_table) + 1)

    return metrics, coef_table


def train_standard_bayesian_lasso(train_data, test_data):
    """Train one Bayesian LASSO model using all regimes together."""
    train_sample = sample_rows(train_data, MAX_GLOBAL_TRAIN_ROWS)
    x_train, x_test, y_train, y_test, _, y_scaler = scale_train_and_test(train_sample, test_data)

    model = BayesianLassoGibbs(
        lambda_value=1.0,
        n_iterations=MCMC_ITERATIONS,
        burn_in=BURN_IN,
        thin=THIN,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = invert_y_scale(model.predict(x_test), y_scaler)
    metrics = calculate_metrics(y_test, predictions, "Standard Bayesian LASSO")

    coef_table = make_coefficient_table(
        "Standard Bayesian LASSO",
        FEATURE_COLUMNS,
        model.coef_,
        model.lower_95_,
        model.upper_95_,
    )

    diagnostics = pd.DataFrame(
        [
            {
                "Model": "Standard Bayesian LASSO",
                "Regime": "All",
                "Rows_Used": len(train_sample),
                "Iterations": MCMC_ITERATIONS,
                "Burn_In": BURN_IN,
                "Thin": THIN,
                **model.diagnostics_,
            }
        ]
    )

    return metrics, coef_table, diagnostics


def train_one_regime_bayesian_lasso(train_data, test_data, regime):
    """Train one Bayesian LASSO model for one regime."""
    regime_train = train_data[train_data["Regime"] == regime].copy()
    regime_test = test_data[test_data["Regime"] == regime].copy()

    if len(regime_train) < 200 or len(regime_test) == 0:
        return None, None, None, None

    train_sample = sample_rows(regime_train, MAX_REGIME_TRAIN_ROWS)
    x_train, x_test, y_train, y_test, _, y_scaler = scale_train_and_test(train_sample, regime_test)

    model = BayesianLassoGibbs(
        lambda_value=1.0,
        n_iterations=MCMC_ITERATIONS,
        burn_in=BURN_IN,
        thin=THIN,
        random_state=RANDOM_STATE + REGIME_ORDER.index(regime),
    )
    model.fit(x_train, y_train)

    predictions = invert_y_scale(model.predict(x_test), y_scaler)

    prediction_table = regime_test[["Date", "Ticker", "Regime", TARGET_COLUMN]].copy()
    prediction_table["Prediction"] = predictions

    coef_table = make_coefficient_table(
        f"Regime Bayesian LASSO - {regime}",
        FEATURE_COLUMNS,
        model.coef_,
        model.lower_95_,
        model.upper_95_,
    )
    coef_table["Regime"] = regime

    diagnostics = pd.DataFrame(
        [
            {
                "Model": f"Regime Bayesian LASSO - {regime}",
                "Regime": regime,
                "Rows_Used": len(train_sample),
                "Iterations": MCMC_ITERATIONS,
                "Burn_In": BURN_IN,
                "Thin": THIN,
                **model.diagnostics_,
            }
        ]
    )

    return prediction_table, coef_table, len(train_sample), diagnostics


def train_regime_adaptive_bayesian_lasso(train_data, test_data):
    """Train separate Bayesian LASSO models inside each HMM regime."""
    prediction_tables = []
    coefficient_tables = []
    diagnostic_tables = []
    training_rows = {}

    for regime in REGIME_ORDER:
        print(f"\nTraining Regime-Adaptive Bayesian LASSO for {regime}")
        predictions, coefficients, rows_used, diagnostics = train_one_regime_bayesian_lasso(
            train_data,
            test_data,
            regime,
        )

        if predictions is None:
            print(f"Skipping {regime}: not enough train or test rows.")
            continue

        prediction_tables.append(predictions)
        coefficient_tables.append(coefficients)
        diagnostic_tables.append(diagnostics)
        training_rows[regime] = rows_used
        print(f"Rows used for training: {rows_used}")

    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    all_coefficients = pd.concat(coefficient_tables, ignore_index=True)
    all_diagnostics = pd.concat(diagnostic_tables, ignore_index=True)

    metrics = calculate_metrics(
        all_predictions[TARGET_COLUMN],
        all_predictions["Prediction"],
        "Regime-Adaptive Bayesian LASSO",
    )

    return metrics, all_coefficients, all_predictions, training_rows, all_diagnostics


def create_selected_feature_table(regime_coefficients):
    """Create the paper-style feature x regime selection table."""
    rows = []

    for feature in FEATURE_COLUMNS:
        row = {"Feature": feature}

        for regime in REGIME_ORDER:
            regime_rows = regime_coefficients[
                (regime_coefficients["Regime"] == regime)
                & (regime_coefficients["Feature"] == feature)
            ]

            if regime_rows.empty:
                row[regime] = "No model"
            else:
                row[regime] = "Yes" if bool(regime_rows["Selected"].iloc[0]) else "No"

        rows.append(row)

    table = pd.DataFrame(rows)

    selection_count = (table[REGIME_ORDER] == "Yes").sum(axis=1)
    table["Selected_Regime_Count"] = selection_count
    table = table.sort_values(
        ["Selected_Regime_Count", "Feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return table


def write_interpretation(
    model_comparison,
    selected_feature_table,
    training_rows,
    split_date,
    mcmc_diagnostics,
):
    """Write a simple interpretation file for the research report."""
    interpretation_file = OUTPUT_DIR / "interpretation.txt"

    best_row = model_comparison.sort_values("RMSE").iloc[0]

    lines = []
    lines.append("BAYESIAN LASSO EXPERIMENT INTERPRETATION")
    lines.append("")
    lines.append(f"Train/test split date: {pd.Timestamp(split_date).date()}")
    lines.append("")
    lines.append("Models compared:")
    lines.append("1. Classical LASSO")
    lines.append("2. Standard Bayesian LASSO")
    lines.append("3. Regime-Adaptive Bayesian LASSO")
    lines.append("")
    lines.append("Model comparison:")
    lines.append(model_comparison.round(6).to_string(index=False))
    lines.append("")
    lines.append(f"Best model by RMSE: {best_row['Model']}")
    lines.append("")
    lines.append("Rows used by each regime-specific Bayesian LASSO:")
    for regime, rows_used in training_rows.items():
        lines.append(f"{regime}: {rows_used}")
    lines.append("")
    lines.append("MCMC diagnostic summary:")
    lines.append(mcmc_diagnostics.round(6).to_string(index=False))
    lines.append("")
    lines.append(
        "Diagnostic meaning: Saved_Samples is the number of retained posterior "
        "draws after burn-in and thinning. ESS is a single-chain effective sample "
        "size estimate. MCSE is the Monte Carlo standard error of posterior mean "
        "coefficients. These diagnostics are lightweight stability checks, not a "
        "replacement for multi-chain R-hat diagnostics."
    )
    lines.append("")
    lines.append("Selected features by regime:")
    lines.append(selected_feature_table.to_string(index=False))
    lines.append("")
    lines.append("Research meaning:")
    lines.append(
        "The standard Bayesian LASSO uses one global coefficient vector for all "
        "market conditions."
    )
    lines.append(
        "The regime-adaptive Bayesian LASSO allows Recovery, Bull, Bear, and Crisis "
        "periods to select different features."
    )
    lines.append(
        "If the selected-feature table differs across regimes, this supports the "
        "idea of regime-dependent sparsity."
    )

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")
    return interpretation_file


def save_outputs(
    model_comparison,
    lasso_coefficients,
    standard_coefficients,
    regime_coefficients,
    selected_feature_table,
    regime_predictions,
    training_rows,
    split_date,
    mcmc_diagnostics,
):
    """Save all experiment outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_comparison_file = OUTPUT_DIR / "model_comparison.csv"
    lasso_file = OUTPUT_DIR / "classical_lasso_coefficients.csv"
    standard_file = OUTPUT_DIR / "standard_bayesian_lasso_coefficients.csv"
    regime_file = OUTPUT_DIR / "regime_adaptive_coefficients_long.csv"
    selected_file = OUTPUT_DIR / "selected_features_by_regime.csv"
    prediction_file = OUTPUT_DIR / "regime_adaptive_predictions.csv"
    diagnostics_file = OUTPUT_DIR / "mcmc_diagnostics.csv"

    model_comparison.to_csv(model_comparison_file, index=False)
    lasso_coefficients.to_csv(lasso_file, index=False)
    standard_coefficients.to_csv(standard_file, index=False)
    regime_coefficients.to_csv(regime_file, index=False)
    selected_feature_table.to_csv(selected_file, index=False)
    regime_predictions.to_csv(prediction_file, index=False)
    mcmc_diagnostics.to_csv(diagnostics_file, index=False)

    interpretation_file = write_interpretation(
        model_comparison,
        selected_feature_table,
        training_rows,
        split_date,
        mcmc_diagnostics,
    )

    return {
        "model_comparison": model_comparison_file,
        "classical_lasso": lasso_file,
        "standard_coefficients": standard_file,
        "regime_coefficients": regime_file,
        "selected_features": selected_file,
        "regime_predictions": prediction_file,
        "mcmc_diagnostics": diagnostics_file,
        "interpretation": interpretation_file,
    }


def main():
    print("STAGE 5: Bayesian LASSO experiment")

    raw_data = load_data()
    data_with_target = add_research_features(raw_data)
    model_data = prepare_model_data(data_with_target)

    train_data, test_data, split_date = make_time_split(model_data)

    print(f"\nRows available: {len(model_data)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Testing rows: {len(test_data)}")
    print(f"Train/test split date: {pd.Timestamp(split_date).date()}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")

    print("\nTraining Classical LASSO baseline")
    lasso_metrics, lasso_coefficients = train_classical_lasso(train_data, test_data)

    print("\nTraining Standard Bayesian LASSO")
    standard_metrics, standard_coefficients, standard_diagnostics = train_standard_bayesian_lasso(
        train_data,
        test_data,
    )

    print("\nTraining Regime-Adaptive Bayesian LASSO")
    (
        adaptive_metrics,
        regime_coefficients,
        regime_predictions,
        training_rows,
        regime_diagnostics,
    ) = train_regime_adaptive_bayesian_lasso(train_data, test_data)

    model_comparison = pd.DataFrame(
        [
            lasso_metrics,
            standard_metrics,
            adaptive_metrics,
        ]
    )

    selected_feature_table = create_selected_feature_table(regime_coefficients)
    mcmc_diagnostics = pd.concat(
        [standard_diagnostics, regime_diagnostics],
        ignore_index=True,
    )

    outputs = save_outputs(
        model_comparison,
        lasso_coefficients,
        standard_coefficients,
        regime_coefficients,
        selected_feature_table,
        regime_predictions,
        training_rows,
        split_date,
        mcmc_diagnostics,
    )

    print("\nMODEL COMPARISON")
    print(model_comparison.round(6).to_string(index=False))

    print("\nSELECTED FEATURES BY REGIME")
    print(selected_feature_table.to_string(index=False))

    print("\nMCMC DIAGNOSTICS")
    print(mcmc_diagnostics.round(6).to_string(index=False))

    print("\nSaved files:")
    for path in outputs.values():
        print(path)

    print("\nDone. This is the first Bayesian LASSO baseline and adaptive experiment.")


if __name__ == "__main__":
    main()
