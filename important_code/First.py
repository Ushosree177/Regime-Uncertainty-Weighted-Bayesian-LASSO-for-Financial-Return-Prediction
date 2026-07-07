"""
Step 1, Step 2, Step 3 market regime pipeline.

Goal:
1. Collect daily OHLCV market data.
2. Create technical indicators from prices and volume.
3. Use a Hidden Markov Model to label market regimes.

The code is intentionally written in a simple style so it is easy to explain.
"""

from pathlib import Path

import numpy as np
import pandas as pd


START_DATE = "2005-01-01"
# yfinance treats the end date as exclusive, so this includes all trading days in 2025.
END_DATE = "2026-01-01"

DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "raw_prices.csv"
FEATURE_FILE = DATA_DIR / "technical_indicators.csv"
REGIME_FILE = DATA_DIR / "market_regimes.csv"
FINAL_FILE = DATA_DIR / "final_dataset_with_regimes.csv"

# 30 large, liquid US stocks from different sectors.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "JPM", "BAC", "V", "MA",
    "JNJ", "PFE", "UNH", "MRK", "KO",
    "PEP", "WMT", "COST", "HD", "MCD",
    "XOM", "CVX", "CAT", "BA", "GE",
    "NFLX", "ADBE", "CSCO", "INTC", "IBM",
]

# SPY is used as a broad market proxy for regime detection.
MARKET_TICKER = "SPY"


def download_market_data(tickers, start_date, end_date):
    """Download daily Open, High, Low, Close, and Volume data."""
    try:
        import yfinance as yf
    except ImportError as error:
        raise ImportError("Please install yfinance: pip install yfinance") from error

    all_tickers = sorted(set(tickers + [MARKET_TICKER]))
    frames = []

    for ticker in all_tickers:
        print(f"Downloading {ticker}...")
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
        )

        if data.empty:
            print(f"Warning: no data found for {ticker}")
            continue

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data.reset_index()
        data["Ticker"] = ticker

        keep_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        frames.append(data[keep_columns])

    if not frames:
        raise ValueError("No data was downloaded. Check internet connection or ticker names.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return prices


def add_technical_indicators(one_stock):
    """Create readable technical indicators for one ticker."""
    df = one_stock.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["Daily_Return"] = close.pct_change()
    df["Log_Return"] = np.log(close / close.shift(1))

    # Trend indicators
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA_50"] = close.ewm(span=50, adjust=False).mean()

    # Momentum indicators
    df["Momentum_10"] = close - close.shift(10)
    df["ROC_10"] = close.pct_change(10)
    df["RSI_14"] = calculate_rsi(close, window=14)

    macd, macd_signal, macd_histogram = calculate_macd(close)
    df["MACD"] = macd
    df["MACD_Signal"] = macd_signal
    df["MACD_Histogram"] = macd_histogram

    # Volatility indicators
    df["True_Range"] = calculate_true_range(high, low, close)
    df["ATR_14"] = df["True_Range"].rolling(14).mean()
    df["Volatility_20"] = df["Daily_Return"].rolling(20).std()

    middle_band = close.rolling(20).mean()
    band_std = close.rolling(20).std()
    df["Bollinger_Middle"] = middle_band
    df["Bollinger_Upper"] = middle_band + 2 * band_std
    df["Bollinger_Lower"] = middle_band - 2 * band_std
    df["Bollinger_Width"] = (df["Bollinger_Upper"] - df["Bollinger_Lower"]) / middle_band
    df["Bollinger_Position"] = (close - df["Bollinger_Lower"]) / (
        df["Bollinger_Upper"] - df["Bollinger_Lower"]
    )

    # Volume indicators
    df["Volume_SMA_20"] = volume.rolling(20).mean()
    df["Volume_Ratio"] = volume / df["Volume_SMA_20"]
    df["OBV"] = calculate_obv(close, volume)

    return df


def calculate_rsi(close, window):
    """Relative Strength Index."""
    change = close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    average_gain = gain.rolling(window).mean()
    average_loss = loss.rolling(window).mean()

    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi


def calculate_macd(close):
    """Moving Average Convergence Divergence."""
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_true_range(high, low, close):
    """Largest of three daily price ranges used in ATR."""
    previous_close = close.shift(1)
    range_1 = high - low
    range_2 = (high - previous_close).abs()
    range_3 = (low - previous_close).abs()

    ranges = pd.concat([range_1, range_2, range_3], axis=1)
    return ranges.max(axis=1)


def calculate_obv(close, volume):
    """On-Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def create_all_indicators(prices):
    """Apply indicators ticker by ticker."""
    feature_frames = []

    for ticker, group in prices.groupby("Ticker"):
        print(f"Creating indicators for {ticker}...")
        feature_frames.append(add_technical_indicators(group))

    features = pd.concat(feature_frames, ignore_index=True)
    features = features.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return features


def detect_market_regimes(features, number_of_regimes=4):
    """Fit HMM on market return, volatility, and volume activity."""
    try:
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:
        message = "Please install hmmlearn and scikit-learn: pip install hmmlearn scikit-learn"
        raise ImportError(message) from error

    market = features[features["Ticker"] == MARKET_TICKER].copy()
    market = market.sort_values("Date").reset_index(drop=True)

    hmm_columns = ["Daily_Return", "Volatility_20", "Volume_Ratio"]
    market = market.dropna(subset=hmm_columns).copy()

    scaler = StandardScaler()
    hmm_input = scaler.fit_transform(market[hmm_columns])

    model = GaussianHMM(
        n_components=number_of_regimes,
        covariance_type="full",
        n_iter=500,
        random_state=42,
    )
    model.fit(hmm_input)
    market["Regime_Number"] = model.predict(hmm_input)

    regime_names = name_regimes(market)
    market["Regime"] = market["Regime_Number"].map(regime_names)

    return market[["Date", "Regime_Number", "Regime"]]


def name_regimes(market):
    """Convert HMM state numbers into simple finance names."""
    summary = (
        market.groupby("Regime_Number")
        .agg(
            Average_Return=("Daily_Return", "mean"),
            Average_Volatility=("Volatility_20", "mean"),
        )
        .reset_index()
    )

    names = {}

    crisis_row = summary.sort_values(
        ["Average_Volatility", "Average_Return"],
        ascending=[False, True],
    ).iloc[0]
    crisis_state = int(crisis_row["Regime_Number"])
    names[crisis_state] = "Crisis"

    remaining = summary[summary["Regime_Number"] != crisis_state].copy()

    bull_row = remaining.sort_values("Average_Return", ascending=False).iloc[0]
    bull_state = int(bull_row["Regime_Number"])
    names[bull_state] = "Bull"

    remaining = remaining[remaining["Regime_Number"] != bull_state].copy()

    bear_state = int(remaining.sort_values("Average_Return").iloc[0]["Regime_Number"])
    names[bear_state] = "Bear"

    for state in remaining["Regime_Number"]:
        if int(state) not in names:
            names[int(state)] = "Recovery"

    return names


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("\nSTEP 1: Collecting market data")
    prices = download_market_data(TICKERS, START_DATE, END_DATE)
    prices.to_csv(RAW_FILE, index=False)
    print(f"Saved raw data: {RAW_FILE}")

    print("\nSTEP 2: Creating technical indicators")
    features = create_all_indicators(prices)
    features.to_csv(FEATURE_FILE, index=False)
    print(f"Saved indicator data: {FEATURE_FILE}")

    print("\nSTEP 3: Detecting market regimes with HMM")
    regimes = detect_market_regimes(features, number_of_regimes=4)
    regimes.to_csv(REGIME_FILE, index=False)
    print(f"Saved regime labels: {REGIME_FILE}")

    final_data = features.merge(regimes, on="Date", how="left")
    final_data = final_data.dropna(subset=["Regime"]).reset_index(drop=True)
    final_data.to_csv(FINAL_FILE, index=False)
    print(f"Saved final dataset: {FINAL_FILE}")

    print("\nDone. The project now covers Step 1, Step 2, and Step 3.")


if __name__ == "__main__":
    main()
