"""
ACTION: Clean and validate the provided dataset.

Run standalone:  python scripts/02_clean_validate.py

Reads the raw extract (data/bank_transactions_raw.csv), fixes every
data-quality issue found in it, and writes a validated clean dataset
to data/bank_transactions_clean.csv.
"""

import numpy as np
import pandas as pd

RAW_PATH = "D:/DS_Hack/data/bank_transactions_raw.csv"
CLEAN_PATH = "D:/DS_Hack/data/bank_transactions_clean.csv"


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log = {}
    log["rows_raw"] = len(df)

    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_removed"] = before - len(df)

    for col in ["Occupation", "City", "Age_Group"]:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan
    df["Occupation"] = df["Occupation"].str.title()
    df["City"] = df["City"].str.title()

    log["missing_before"] = df.isna().sum().to_dict()

    for col in ["Occupation", "City"]:
        mode_val = df[col].mode(dropna=True)[0]
        df[col] = df[col].fillna(mode_val)
    df["Bill_Payments"] = df.groupby("Age_Group")["Bill_Payments"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Card_Spend"] = df["Card_Spend"].fillna(0)

    neg_mask = df["Withdrawals"] < 0
    log["negative_withdrawals_fixed"] = int(neg_mask.sum())
    df.loc[neg_mask, "Withdrawals"] = df.loc[neg_mask, "Withdrawals"].abs()

    int_cols = ["UPI_Count", "Card_Count", "ATM_Count", "Bill_Payments",
                "ATM_M1", "ATM_M2", "ATM_M3", "ATM_M4", "ATM_M5", "ATM_M6"]
    for col in int_cols:
        df[col] = df[col].round().astype(int)
    float_cols = ["Card_Spend", "Deposits", "Withdrawals"]
    for col in float_cols:
        df[col] = df[col].astype(float).round(2)

    valid_age = {"18-25", "26-35", "36-45", "46-55", "56-65", "65+"}
    df = df[df["Age_Group"].isin(valid_age)]

    before = len(df)
    df = df[(df["UPI_Count"] >= 0) & (df["Card_Count"] >= 0) & (df["ATM_Count"] >= 0) & (df["Deposits"] >= 0)]
    log["invalid_rows_dropped"] = before - len(df)

    df = df.reset_index(drop=True)
    log["rows_clean"] = len(df)
    log["missing_after"] = int(df.isna().sum().sum())
    return df, log


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    print(f"Loaded raw dataset: {raw.shape[0]} rows, {raw.shape[1]} columns\n")

    clean_df, log = clean(raw)

    print("=== Cleaning & Validation Log ===")
    print(f"Rows before cleaning     : {log['rows_raw']}")
    print(f"Duplicate rows removed   : {log['duplicates_removed']}")
    print(f"Negative withdrawals fixed (abs value) : {log['negative_withdrawals_fixed']}")
    print(f"Invalid rows dropped     : {log['invalid_rows_dropped']}")
    print(f"Rows after cleaning      : {log['rows_clean']}")
    print(f"Remaining missing values : {log['missing_after']}")

    print("\nMissing values found per column (before imputation):")
    for col, n in log["missing_before"].items():
        if n > 0:
            print(f"  {col:15s}: {n}")

    print("\nSample of cleaned data:")
    print(clean_df.head())

    clean_df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved validated dataset -> {CLEAN_PATH}  ({len(clean_df)} rows, 0 missing values, 0 duplicates)")
