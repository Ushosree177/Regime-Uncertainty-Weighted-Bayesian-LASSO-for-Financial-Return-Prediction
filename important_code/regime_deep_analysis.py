"""
Day 1 to Day 5 regime validation and interpretation.

This script does the next research checks after Step 1, Step 2, and Step 3:

Day 1: Plot SPY price with regime colors.
Day 2: Refit HMM and print the transition matrix.
Day 3: Compute indicator statistics by regime.
Day 4: Create clean regime-indicator comparison tables.
Day 5: Write a simple interpretation that explains what the results mean.

The code is written in a simple way so it is easy to explain in a meeting.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "regime_deep_analysis"
FINAL_FILE = DATA_DIR / "final_dataset_with_regimes.csv"

MARKET_TICKER = "SPY"
NUMBER_OF_REGIMES = 4

REGIME_COLORS = {
    "Bull": "green",
    "Bear": "orange",
    "Crisis": "red",
    "Recovery": "blue",
}

IMPORTANT_INDICATORS = [
    "RSI_14",
    "MACD",
    "MACD_Histogram",
    "EMA_20",
    "ATR_14",
    "Volatility_20",
    "Volume_Ratio",
    "Bollinger_Width",
    "Bollinger_Position",
    "OBV",
]

COMPARABLE_INDICATORS = [
    "RSI_14",
    "MACD_Histogram",
    "Normalized_ATR",
    "Volatility_20",
    "Volume_Ratio",
    "Bollinger_Width",
    "Bollinger_Position",
    "Daily_Return",
]


def load_market_data():
    """Load final dataset and keep only SPY rows for market regime analysis."""
    df = pd.read_csv(FINAL_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Normalized_ATR"] = df["ATR_14"] / df["Close"]

    market = df[df["Ticker"] == MARKET_TICKER].copy()
    market = market.sort_values("Date").reset_index(drop=True)
    return df, market


def plot_price_by_regime(market):
    """Day 1: Plot SPY close price and color each point by HMM regime."""
    plt.figure(figsize=(14, 7))
    plt.plot(market["Date"], market["Close"], color="black", linewidth=1, label="SPY Close")

    for regime, color in REGIME_COLORS.items():
        regime_data = market[market["Regime"] == regime]
        plt.scatter(
            regime_data["Date"],
            regime_data["Close"],
            s=8,
            color=color,
            label=regime,
            alpha=0.75,
        )

    plt.title("SPY Price Colored by HMM Market Regime")
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.legend()
    plt.tight_layout()

    output_file = OUTPUT_DIR / "day1_spy_price_by_regime.png"
    plt.savefig(output_file, dpi=200)
    plt.close()
    return output_file


def plot_crisis_zoom(market, start_date, end_date, file_name, title):
    """Make a zoomed plot for important crisis windows."""
    window = market[(market["Date"] >= start_date) & (market["Date"] <= end_date)].copy()

    plt.figure(figsize=(12, 6))
    plt.plot(window["Date"], window["Close"], color="black", linewidth=1, label="SPY Close")

    for regime, color in REGIME_COLORS.items():
        regime_data = window[window["Regime"] == regime]
        plt.scatter(
            regime_data["Date"],
            regime_data["Close"],
            s=25,
            color=color,
            label=regime,
            alpha=0.85,
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("SPY Close Price")
    plt.legend()
    plt.tight_layout()

    output_file = OUTPUT_DIR / file_name
    plt.savefig(output_file, dpi=200)
    plt.close()
    return output_file


def fit_hmm_again(market):
    """Day 2: Refit HMM so we can read the transition matrix."""
    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    hmm_columns = ["Daily_Return", "Volatility_20", "Volume_Ratio"]
    hmm_data = market.dropna(subset=hmm_columns).copy()

    scaler = StandardScaler()
    hmm_input = scaler.fit_transform(hmm_data[hmm_columns])

    model = GaussianHMM(
        n_components=NUMBER_OF_REGIMES,
        covariance_type="full",
        n_iter=500,
        random_state=42,
    )
    model.fit(hmm_input)

    hmm_data["New_Regime_Number"] = model.predict(hmm_input)
    state_names = map_hmm_states_to_existing_regime_names(hmm_data)

    transition_matrix = pd.DataFrame(model.transmat_)
    transition_matrix.index = [state_names[i] for i in range(NUMBER_OF_REGIMES)]
    transition_matrix.columns = [state_names[i] for i in range(NUMBER_OF_REGIMES)]

    return transition_matrix


def map_hmm_states_to_existing_regime_names(hmm_data):
    """
    HMM state numbers can change when fitting.
    This maps new state numbers to the existing labels by majority vote.
    """
    state_names = {}

    for state in sorted(hmm_data["New_Regime_Number"].unique()):
        rows = hmm_data[hmm_data["New_Regime_Number"] == state]
        most_common_label = rows["Regime"].value_counts().idxmax()
        state_names[int(state)] = most_common_label

    return state_names


def calculate_regime_statistics(market):
    """Day 3: Calculate market return and indicator averages by regime."""
    columns = ["Daily_Return"] + IMPORTANT_INDICATORS
    available_columns = [column for column in columns if column in market.columns]

    regime_stats = (
        market.groupby("Regime")[available_columns]
        .agg(["mean", "std"])
        .round(6)
    )

    return regime_stats


def create_indicator_table(market):
    """Day 4: Create a clean table with regimes as columns."""
    available_indicators = [
        indicator for indicator in IMPORTANT_INDICATORS if indicator in market.columns
    ]

    table = market.groupby("Regime")[available_indicators].mean().T
    table = table.round(6)

    useful_order = ["Bull", "Bear", "Crisis", "Recovery"]
    existing_order = [regime for regime in useful_order if regime in table.columns]
    table = table[existing_order]

    return table


def create_cross_stock_indicator_table(full_data):
    """
    Day 4 extra table.

    This uses all stocks, not only SPY.
    We avoid raw price-level indicators like EMA and OBV here because those depend
    strongly on stock price level and company size.
    """
    table = full_data.groupby("Regime")[COMPARABLE_INDICATORS].mean().T
    table = table.round(6)

    useful_order = ["Bull", "Bear", "Crisis", "Recovery"]
    existing_order = [regime for regime in useful_order if regime in table.columns]
    table = table[existing_order]

    return table


def create_rank_table(indicator_table):
    """Rank regimes for each indicator from highest value to lowest value."""
    rank_rows = []

    for indicator in indicator_table.index:
        sorted_values = indicator_table.loc[indicator].sort_values(ascending=False)
        rank_rows.append(
            {
                "Indicator": indicator,
                "Highest_Regime": sorted_values.index[0],
                "Highest_Value": sorted_values.iloc[0],
                "Lowest_Regime": sorted_values.index[-1],
                "Lowest_Value": sorted_values.iloc[-1],
            }
        )

    return pd.DataFrame(rank_rows)


def write_interpretation(regime_stats, indicator_table, transition_matrix):
    """Day 5: Create a simple human-readable explanation."""
    crisis_volatility = regime_stats.loc["Crisis", ("Daily_Return", "std")]
    recovery_volatility = regime_stats.loc["Recovery", ("Daily_Return", "std")]

    atr_highest_regime = indicator_table.loc["ATR_14"].sort_values(ascending=False).index[0]
    volume_highest_regime = indicator_table.loc["Volume_Ratio"].sort_values(ascending=False).index[0]
    rsi_highest_regime = indicator_table.loc["RSI_14"].sort_values(ascending=False).index[0]

    report = []

    report.append("DAY 5 INTERPRETATION\n")
    report.append("1. The HMM regimes are meaningful because volatility changes clearly by regime.")
    report.append(
        f"   Crisis volatility is {crisis_volatility:.6f}, while Recovery volatility is "
        f"{recovery_volatility:.6f}."
    )
    report.append("")

    report.append("2. The transition matrix tells us whether regimes are persistent.")
    report.append(
        "   Large diagonal values mean the market usually stays in the same regime "
        "from one day to the next."
    )
    report.append("")
    report.append(transition_matrix.round(4).to_string())
    report.append("")

    report.append("3. Indicator behavior changes across regimes.")
    report.append(f"   ATR is highest in: {atr_highest_regime}")
    report.append(f"   Volume Ratio is highest in: {volume_highest_regime}")
    report.append(f"   RSI is highest in: {rsi_highest_regime}")
    report.append("")

    report.append("4. Research meaning:")
    report.append(
        "   If ATR, volume, RSI, MACD, and volatility behave differently in different "
        "regimes, then one fixed feature-selection model is too rigid."
    )
    report.append(
        "   This supports the next idea: Regime-Adaptive Bayesian LASSO, where each "
        "market regime can have its own important indicators."
    )

    return "\n".join(report)


def save_table(table, file_name):
    """Save a table as CSV and return the path."""
    output_file = OUTPUT_DIR / file_name
    table.to_csv(output_file)
    return output_file


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    full_data, market = load_market_data()

    print("\nDAY 1: Creating regime plots")
    plot_1 = plot_price_by_regime(market)
    plot_2 = plot_crisis_zoom(
        market,
        "2008-09-01",
        "2009-03-31",
        "day1_2008_crisis_zoom.png",
        "2008 Financial Crisis: SPY Colored by Regime",
    )
    plot_3 = plot_crisis_zoom(
        market,
        "2020-02-15",
        "2020-04-30",
        "day1_covid_crash_zoom.png",
        "COVID Crash: SPY Colored by Regime",
    )
    print(f"Saved: {plot_1}")
    print(f"Saved: {plot_2}")
    print(f"Saved: {plot_3}")

    print("\nDAY 2: Creating HMM transition matrix")
    transition_matrix = fit_hmm_again(market)
    transition_file = save_table(transition_matrix.round(6), "day2_transition_matrix.csv")
    print(transition_matrix.round(4))
    print(f"Saved: {transition_file}")

    print("\nDAY 3: Calculating regime statistics")
    regime_stats = calculate_regime_statistics(market)
    stats_file = save_table(regime_stats, "day3_regime_statistics.csv")
    print(regime_stats)
    print(f"Saved: {stats_file}")

    print("\nDAY 4: Creating indicator comparison table")
    indicator_table = create_indicator_table(market)
    cross_stock_table = create_cross_stock_indicator_table(full_data)
    rank_table = create_rank_table(indicator_table)
    indicator_file = save_table(indicator_table, "day4_indicator_table.csv")
    cross_stock_file = save_table(
        cross_stock_table,
        "day4_cross_stock_indicator_table.csv",
    )
    rank_file = save_table(rank_table, "day4_indicator_rank_table.csv")
    print(indicator_table)
    print(f"Saved: {indicator_file}")
    print("\nCross-stock comparable indicator table:")
    print(cross_stock_table)
    print(f"Saved: {cross_stock_file}")
    print(f"Saved: {rank_file}")

    print("\nDAY 5: Writing interpretation")
    interpretation = write_interpretation(regime_stats, indicator_table, transition_matrix)
    interpretation_file = OUTPUT_DIR / "day5_interpretation.txt"
    interpretation_file.write_text(interpretation, encoding="utf-8")
    print(interpretation)
    print(f"Saved: {interpretation_file}")


if __name__ == "__main__":
    main()
