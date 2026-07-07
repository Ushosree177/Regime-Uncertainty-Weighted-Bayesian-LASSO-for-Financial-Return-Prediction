"""
Second market validation using sector ETFs.

Purpose:

The main real-data experiment uses a basket of individual US equities. This
script repeats the core validation on a different asset universe: sector ETFs.

This checks whether the conclusions are tied only to the original stock basket.

Dataset:

    SPY as market proxy for HMM regime detection
    Sector ETFs as prediction universe

Models compared:

1. Classical LASSO
2. Elastic Net
3. Adaptive LASSO
4. Regime-specific LASSO
5. Standard Bayesian LASSO
6. Regime-specific Bayesian LASSO
7. Proposed Prior - Full
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from First import add_technical_indicators, detect_market_regimes
from robust_real_data_validation import (
    TARGET_COLUMN,
    add_research_features,
    calculate_feature_evidence,
    calculate_lambda_matrix,
    calculate_regime_uncertainty,
    prepare_model_data,
    split_train_validation_test,
    train_adaptive_lasso,
    train_elastic_net,
    train_lasso,
    train_regime_bayesian_model,
    train_regime_lasso,
    train_standard_bayesian_lasso,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "second_market_validation"

START_DATE = "2010-01-01"
END_DATE = "2026-01-01"
MARKET_TICKER = "SPY"

SECTOR_ETFS = [
    "XLB",  # Materials
    "XLC",  # Communication Services
    "XLE",  # Energy
    "XLF",  # Financials
    "XLI",  # Industrials
    "XLK",  # Technology
    "XLP",  # Consumer Staples
    "XLRE", # Real Estate
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLY",  # Consumer Discretionary
]


def download_prices():
    """Download OHLCV data using yfinance."""
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError("Please install yfinance before running this script.") from error

    tickers = sorted(set(SECTOR_ETFS + [MARKET_TICKER]))
    frames = []

    for ticker in tickers:
        print(f"Downloading {ticker}")
        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            print(f"Warning: no data for {ticker}")
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Ticker"] = ticker
        keep_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        frames.append(data[keep_columns])

    if not frames:
        raise ValueError("No ETF data was downloaded.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return prices


def create_indicators(prices):
    """Create the same technical indicators ticker by ticker."""
    frames = []

    for ticker, group in prices.groupby("Ticker"):
        print(f"Creating indicators for {ticker}")
        frames.append(add_technical_indicators(group))

    features = pd.concat(frames, ignore_index=True)
    features = features.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return features


def create_sector_dataset():
    """Download, create indicators, detect regimes, and merge final dataset."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw_file = OUTPUT_DIR / "sector_etf_raw_prices.csv"
    indicator_file = OUTPUT_DIR / "sector_etf_indicators.csv"
    regime_file = OUTPUT_DIR / "sector_etf_market_regimes.csv"
    final_file = OUTPUT_DIR / "sector_etf_final_dataset_with_regimes.csv"

    if final_file.exists():
        print(f"Using existing dataset: {final_file}")
        return pd.read_csv(final_file)

    prices = download_prices()
    prices.to_csv(raw_file, index=False)

    features = create_indicators(prices)
    features.to_csv(indicator_file, index=False)

    regimes = detect_market_regimes(features, number_of_regimes=4)
    regimes.to_csv(regime_file, index=False)

    final_data = features.merge(regimes, on="Date", how="left")
    final_data = final_data.dropna(subset=["Regime"]).reset_index(drop=True)
    final_data.to_csv(final_file, index=False)

    return final_data


def run_model_comparison(final_data):
    """Run the same core model comparison on the ETF universe."""
    final_data["Date"] = pd.to_datetime(final_data["Date"])

    data_with_features = add_research_features(final_data)
    model_data = prepare_model_data(data_with_features)

    # Evaluate only sector ETFs. SPY was used as the market regime proxy.
    model_data = model_data[model_data["Ticker"] != MARKET_TICKER].copy()

    train_data, validation_data, test_data, train_date, validation_date = split_train_validation_test(
        model_data
    )
    final_train = pd.concat([train_data, validation_data], ignore_index=True)

    print(f"\nRows available: {len(model_data)}")
    print(f"Training rows: {len(final_train)}")
    print(f"Testing rows: {len(test_data)}")
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

    print("Training Proposed Prior - Full")
    full_lambda = calculate_lambda_matrix(uncertainty, evidence, "full")
    full_lambda = full_lambda.clip(lower=0.05, upper=20.0)
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

    return model_results, selected_features, uncertainty, evidence


def save_outputs(model_results, selected_features, uncertainty, evidence):
    """Save all second-market outputs."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    model_file = OUTPUT_DIR / "sector_etf_model_comparison.csv"
    selected_file = OUTPUT_DIR / "sector_etf_selected_features.csv"
    uncertainty_file = OUTPUT_DIR / "sector_etf_uncertainty_scores.csv"
    evidence_file = OUTPUT_DIR / "sector_etf_feature_evidence.csv"
    interpretation_file = OUTPUT_DIR / "sector_etf_interpretation.txt"

    model_results.round(6).to_csv(model_file, index=False)
    selected_features.to_csv(selected_file, index=False)
    uncertainty.round(6).to_csv(uncertainty_file, index=False)
    evidence.round(6).to_csv(evidence_file)

    lines = []
    lines.append("SECOND MARKET VALIDATION: SECTOR ETF UNIVERSE")
    lines.append("")
    lines.append("Model comparison:")
    lines.append(model_results.round(6).to_string(index=False))
    lines.append("")
    lines.append("Research meaning:")
    lines.append(
        "This experiment checks whether conclusions from the individual-stock "
        "dataset also appear in a different asset universe."
    )
    lines.append(
        "If simple sparse models still perform best, it supports the idea that "
        "real market datasets often have weaker regime-dependent feature "
        "structure than the strong simulation scenario."
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
    print("SECOND MARKET VALIDATION")
    print("Dataset: US sector ETFs")

    final_data = create_sector_dataset()
    model_results, selected_features, uncertainty, evidence = run_model_comparison(final_data)
    outputs = save_outputs(model_results, selected_features, uncertainty, evidence)

    print("\nSECTOR ETF MODEL COMPARISON")
    print(model_results.round(6).to_string(index=False))

    print("\nSaved files:")
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
