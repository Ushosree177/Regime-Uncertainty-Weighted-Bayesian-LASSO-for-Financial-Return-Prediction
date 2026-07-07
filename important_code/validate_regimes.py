"""
Validate HMM market regimes after Step 1, Step 2, and Step 3.

This script answers:
1. What are the first rows of market_regimes.csv?
2. How many unique regimes did HMM find?
3. What are the mean return and volatility of each regime?
4. Did crisis periods like 2008 and March 2020 get separated?
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
REGIME_FILE = DATA_DIR / "market_regimes.csv"
FINAL_FILE = DATA_DIR / "final_dataset_with_regimes.csv"
REPORT_FILE = DATA_DIR / "regime_validation_report.txt"
MARKET_TICKER = "SPY"


def line(title):
    return f"\n{'=' * 70}\n{title}\n{'=' * 70}\n"


def format_table(dataframe):
    return dataframe.to_string(index=True)


def main():
    regimes = pd.read_csv(REGIME_FILE)
    final_data = pd.read_csv(FINAL_FILE)

    regimes["Date"] = pd.to_datetime(regimes["Date"])
    final_data["Date"] = pd.to_datetime(final_data["Date"])

    market = final_data[final_data["Ticker"] == MARKET_TICKER].copy()
    market = market.sort_values("Date").reset_index(drop=True)

    report = []

    report.append(line("TASK A: First 20 Rows of market_regimes.csv"))
    report.append(format_table(regimes.head(20)))

    report.append(line("TASK B: Regime Counts"))
    regime_counts = regimes["Regime"].value_counts()
    report.append(format_table(regime_counts.to_frame("Number_of_Days")))

    report.append(line("Number of Unique Regimes"))
    report.append(str(regimes["Regime"].nunique()))

    report.append(line("TASK C: Regime Return Statistics Using SPY"))
    regime_stats = (
        market.groupby("Regime")
        .agg(
            Mean_Return=("Daily_Return", "mean"),
            Volatility=("Daily_Return", "std"),
            Number_of_Days=("Daily_Return", "count"),
        )
        .sort_values("Volatility", ascending=False)
    )
    report.append(format_table(regime_stats))

    report.append(line("2008 Financial Crisis Check"))
    crisis_2008 = market[
        (market["Date"] >= "2008-09-01") & (market["Date"] <= "2009-03-31")
    ]
    report.append(format_table(crisis_2008["Regime"].value_counts().to_frame("Number_of_Days")))

    report.append(line("COVID Crash Check"))
    covid_crash = market[
        (market["Date"] >= "2020-02-15") & (market["Date"] <= "2020-04-30")
    ]
    report.append(format_table(covid_crash["Regime"].value_counts().to_frame("Number_of_Days")))

    report.append(line("Simple Interpretation Guide"))
    report.append(
        "A useful HMM result should usually show one regime with high volatility "
        "and weak/negative mean return. That regime is a natural Crisis or Bear regime.\n"
        "A Bull regime should usually have positive mean return and lower volatility.\n"
        "If March 2020 and late 2008 mostly fall into the high-volatility regime, "
        "then the HMM is giving meaningful regime labels."
    )

    output = "\n".join(report)
    REPORT_FILE.write_text(output, encoding="utf-8")
    print(output)
    print(f"\nSaved report to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
