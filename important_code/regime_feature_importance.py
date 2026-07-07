"""
Stage 4: Regime-wise feature importance analysis.

This script answers the next research question:

    Do different market regimes need different technical indicators?

We already have HMM regime labels in:

    data/final_dataset_with_regimes.csv

Here we:
1. Create a 5-day future return target for every stock.
2. Split the dataset into Bull, Bear, Crisis, and Recovery rows.
3. Train Random Forest and XGBoost separately inside each regime.
4. Extract feature importance for each regime.
5. Optionally calculate SHAP importance if the shap package is installed.
6. Save clean comparison tables for the research presentation.


"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "final_dataset_with_regimes.csv"
OUTPUT_DIR = DATA_DIR / "regime_feature_importance"

TARGET_COLUMN = "Future_5_Day_Return"
RANDOM_STATE = 42

# This keeps the script fast and avoids one regime dominating the results.
# You can increase this later if your laptop runs it comfortably.
MAX_ROWS_PER_REGIME = 8000

REGIME_ORDER = ["Recovery", "Bull", "Bear", "Crisis"]


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


def load_data():
    """Load the final HMM dataset."""
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def add_research_features(df):
    """
    Add features that are safer for cross-stock modelling.

    Raw moving averages and raw OBV depend heavily on stock price level and company
    size. For feature selection, relative versions are easier to compare.
    """
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
    """Keep only rows where all selected features and the target are available."""
    available_features = [column for column in FEATURE_COLUMNS if column in df.columns]
    model_columns = ["Date", "Ticker", "Regime", TARGET_COLUMN] + available_features

    model_data = df[model_columns].copy()
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=[TARGET_COLUMN] + available_features)

    return model_data, available_features


def sample_one_regime(regime_data):
    """
    Use a fixed maximum number of rows per regime.

    This makes Bull, Bear, Crisis, and Recovery more comparable and keeps the
    script easy to run on a normal laptop.
    """
    if len(regime_data) <= MAX_ROWS_PER_REGIME:
        return regime_data.copy()

    return regime_data.sample(MAX_ROWS_PER_REGIME, random_state=RANDOM_STATE).copy()


def train_random_forest(x_train, y_train):
    """Train a Random Forest regression model."""
    model = RandomForestRegressor(
        n_estimators=180,
        min_samples_leaf=8,
        max_features="sqrt",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    return model


def train_xgboost(x_train, y_train):
    """Train an XGBoost regression model."""
    try:
        from xgboost import XGBRegressor
    except ImportError as error:
        raise ImportError("Please install xgboost: pip install xgboost") from error

    model = XGBRegressor(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def make_importance_table(regime, method, features, importance_values):
    """Convert feature importance values into a clean ranked table."""
    table = pd.DataFrame(
        {
            "Regime": regime,
            "Method": method,
            "Feature": features,
            "Importance": importance_values,
        }
    )

    total = table["Importance"].sum()
    if total > 0:
        table["Normalized_Importance"] = table["Importance"] / total
    else:
        table["Normalized_Importance"] = 0

    table = table.sort_values("Importance", ascending=False).reset_index(drop=True)
    table["Rank"] = np.arange(1, len(table) + 1)
    return table


def calculate_shap_importance(model, x_sample, regime, features):
    """
    Calculate SHAP importance if shap is installed.

    SHAP is optional because it is a heavier package. If it is not installed,
    the rest of the analysis still runs.
    """
    try:
        import shap
    except ImportError:
        return None

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return make_importance_table(regime, "SHAP", features, mean_abs_shap)


def evaluate_model(model, x_test, y_test, regime, method):
    """Save simple prediction quality numbers for transparency."""
    predictions = model.predict(x_test)

    return {
        "Regime": regime,
        "Method": method,
        "Rows_Tested": len(y_test),
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }


def run_models_for_regime(model_data, regime, features):
    """Train RF and XGBoost inside one regime and collect importances."""
    regime_data = model_data[model_data["Regime"] == regime].copy()
    regime_data = sample_one_regime(regime_data)

    x = regime_data[features]
    y = regime_data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    importance_tables = []
    score_rows = []

    print(f"\nTraining models for {regime}")
    print(f"Rows used: {len(regime_data)}")

    random_forest = train_random_forest(x_train, y_train)
    rf_table = make_importance_table(
        regime,
        "Random Forest",
        features,
        random_forest.feature_importances_,
    )
    importance_tables.append(rf_table)
    score_rows.append(evaluate_model(random_forest, x_test, y_test, regime, "Random Forest"))

    xgboost = train_xgboost(x_train, y_train)
    xgb_table = make_importance_table(
        regime,
        "XGBoost",
        features,
        xgboost.feature_importances_,
    )
    importance_tables.append(xgb_table)
    score_rows.append(evaluate_model(xgboost, x_test, y_test, regime, "XGBoost"))

    shap_sample = x_test.sample(min(1000, len(x_test)), random_state=RANDOM_STATE)
    shap_table = calculate_shap_importance(xgboost, shap_sample, regime, features)
    if shap_table is not None:
        importance_tables.append(shap_table)
    else:
        print("SHAP is not installed, so SHAP importance is skipped.")

    return importance_tables, score_rows


def create_top_feature_table(all_importance):
    """
    Create the main presentation table:

        Rank | Recovery | Bull | Bear | Crisis
    """
    tables = []

    for regime in REGIME_ORDER:
        regime_rows = all_importance[all_importance["Regime"] == regime].copy()

        averaged = (
            regime_rows.groupby("Feature")["Normalized_Importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        averaged["Rank"] = np.arange(1, len(averaged) + 1)
        averaged = averaged[["Rank", "Feature"]].head(10)
        averaged = averaged.rename(columns={"Feature": regime})
        tables.append(averaged)

    final_table = tables[0]
    for table in tables[1:]:
        final_table = final_table.merge(table, on="Rank", how="outer")

    return final_table.sort_values("Rank")


def label_importance(value):
    """Convert a numeric importance value into a simple research label."""
    if value >= 0.12:
        return "Very High"
    if value >= 0.07:
        return "High"
    if value >= 0.035:
        return "Medium"
    if value > 0:
        return "Low"
    return "None"


def create_feature_strength_table(all_importance):
    """
    Create a feature x regime table with labels like Low, Medium, High.

    This is useful for explaining whether feature relevance changes by regime.
    """
    average_table = (
        all_importance.groupby(["Feature", "Regime"])["Normalized_Importance"]
        .mean()
        .reset_index()
    )

    pivot = average_table.pivot(
        index="Feature",
        columns="Regime",
        values="Normalized_Importance",
    ).fillna(0)

    existing_order = [regime for regime in REGIME_ORDER if regime in pivot.columns]
    pivot = pivot[existing_order]

    strength_table = pivot.map(label_importance)
    strength_table["Average_Importance"] = pivot.mean(axis=1)
    strength_table = strength_table.sort_values("Average_Importance", ascending=False)

    return strength_table


def save_outputs(all_importance, model_scores):
    """Save all final result files."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_importance_file = OUTPUT_DIR / "all_feature_importance_long.csv"
    score_file = OUTPUT_DIR / "model_scores.csv"
    top_feature_file = OUTPUT_DIR / "top_features_by_regime.csv"
    strength_file = OUTPUT_DIR / "feature_strength_by_regime.csv"

    all_importance.to_csv(all_importance_file, index=False)
    model_scores.to_csv(score_file, index=False)

    top_feature_table = create_top_feature_table(all_importance)
    strength_table = create_feature_strength_table(all_importance)

    top_feature_table.to_csv(top_feature_file, index=False)
    strength_table.to_csv(strength_file)

    return {
        "all_importance": all_importance_file,
        "model_scores": score_file,
        "top_features": top_feature_file,
        "feature_strength": strength_file,
        "top_feature_table": top_feature_table,
        "strength_table": strength_table,
    }


def write_interpretation(top_feature_table, strength_table):
    """Write a short explanation that can be used in the report."""
    interpretation_file = OUTPUT_DIR / "interpretation.txt"

    lines = []
    lines.append("REGIME-WISE FEATURE IMPORTANCE INTERPRETATION")
    lines.append("")
    lines.append("Research question:")
    lines.append("Do different market regimes need different technical indicators?")
    lines.append("")
    lines.append("Top features by regime:")
    lines.append(top_feature_table.to_string(index=False))
    lines.append("")
    lines.append("Feature strength labels:")
    lines.append(strength_table.drop(columns=["Average_Importance"]).to_string())
    lines.append("")
    lines.append("How to use this result:")
    lines.append(
        "If the top features are not the same across Recovery, Bull, Bear, and Crisis, "
        "then feature relevance is regime-dependent."
    )
    lines.append(
        "That supports the next step: Regime-Adaptive Bayesian LASSO, where each "
        "regime can have its own shrinkage strength."
    )

    interpretation_file.write_text("\n".join(lines), encoding="utf-8")
    return interpretation_file


def main():
    print("STAGE 4: Regime-wise feature importance analysis")

    raw_data = load_data()
    data_with_target = add_research_features(raw_data)
    model_data, features = prepare_model_data(data_with_target)

    print(f"\nRows available for modelling: {len(model_data)}")
    print(f"Number of features used: {len(features)}")
    print("Target variable: 5-day future return")

    all_tables = []
    all_scores = []

    for regime in REGIME_ORDER:
        if regime not in model_data["Regime"].unique():
            print(f"\nSkipping {regime}: no rows found.")
            continue

        importance_tables, score_rows = run_models_for_regime(model_data, regime, features)
        all_tables.extend(importance_tables)
        all_scores.extend(score_rows)

    all_importance = pd.concat(all_tables, ignore_index=True)
    model_scores = pd.DataFrame(all_scores)

    outputs = save_outputs(all_importance, model_scores)
    interpretation_file = write_interpretation(
        outputs["top_feature_table"],
        outputs["strength_table"],
    )

    print("\nMAIN TABLE: Top features by regime")
    print(outputs["top_feature_table"].to_string(index=False))

    print("\nSaved files:")
    print(outputs["all_importance"])
    print(outputs["model_scores"])
    print(outputs["top_features"])
    print(outputs["feature_strength"])
    print(interpretation_file)

    print("\nDone. This is the bridge between HMM regimes and Bayesian LASSO.")


if __name__ == "__main__":
    main()
