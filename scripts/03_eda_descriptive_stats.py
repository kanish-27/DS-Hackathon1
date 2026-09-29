"""
ACTION: Perform exploratory data analysis using suitable descriptive statistics.

Run standalone:  python scripts/03_eda_descriptive_stats.py
(requires data/bank_transactions_clean.csv from 02_clean_validate.py)

Engineers the analysis features every later script depends on
(Digital_Ratio, ATM_Trend_Slope, Total_Txn_Value, High_Value_Outlier)
and prints descriptive statistics for the whole customer base.
Saves the enriched dataset to data/bank_transactions_features.csv.
"""

import numpy as np
import pandas as pd

CLEAN_PATH = "D:/DS_Hack/data/bank_transactions_clean.csv"
FEATURES_PATH = "D:/DS_Hack/data/bank_transactions_features.csv"

pd.set_option("display.width", 140)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Digital_Count"] = df["UPI_Count"] + df["Card_Count"]
    df["Total_Channel_Txns"] = df["Digital_Count"] + df["ATM_Count"]
    df["Digital_Ratio"] = (df["Digital_Count"] / df["Total_Channel_Txns"].replace(0, np.nan)).fillna(0)

    months = np.arange(1, 7)
    atm_month_cols = [f"ATM_M{i}" for i in range(1, 7)]
    x = months - months.mean()
    atm_matrix = df[atm_month_cols].to_numpy()
    df["ATM_Trend_Slope"] = (atm_matrix * x).sum(axis=1) / (x**2).sum()

    df["Total_Txn_Value"] = df["Deposits"] + df["Withdrawals"] + df["Card_Spend"]
    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    q1, q3 = df["Total_Txn_Value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    df["High_Value_Outlier"] = df["Total_Txn_Value"] > upper_fence
    return df


if __name__ == "__main__":
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded clean dataset: {df.shape[0]} customers\n")

    df = add_features(df)
    df = flag_outliers(df)

    print("=== Engineered features added ===")
    print("  Digital_Ratio     : share of UPI+Card transactions out of all channel transactions")
    print("  ATM_Trend_Slope   : linear trend of ATM usage across the 6 months (negative = declining)")
    print("  Total_Txn_Value   : Deposits + Withdrawals + Card_Spend (6-month total)")
    print("  High_Value_Outlier: flagged via IQR rule on Total_Txn_Value")

    numeric_cols = ["UPI_Count", "Card_Count", "ATM_Count", "Deposits", "Withdrawals",
                     "Bill_Payments", "Digital_Ratio", "ATM_Trend_Slope", "Total_Txn_Value"]

    print("\n=== Descriptive Statistics (full customer base, n={}) ===".format(len(df)))
    print(df[numeric_cols].describe().round(2).T)

    print("\n=== Skewness (right-skew > 0 means a long high-value tail) ===")
    print(df[numeric_cols].skew().round(2))

    print("\n=== Categorical distribution ===")
    for col in ["Age_Group", "Occupation"]:
        print(f"\n{col}:")
        pct = (df[col].value_counts(normalize=True) * 100).round(1)
        for label, value in pct.items():
            print(f"  {label:20s}: {value:.1f}%")

    df.to_csv(FEATURES_PATH, index=False)
    print(f"\nSaved feature-enriched dataset -> {FEATURES_PATH}")
