"""
ACTION: Identify unusually high transaction patterns requiring further investigation.

Run standalone:  python scripts/08_outlier_detection.py
(requires data/bank_transactions_features.csv from 03_eda_descriptive_stats.py,
 which already computes the High_Value_Outlier flag via the IQR rule)
"""

import pandas as pd

FEATURES_PATH = "D:/DS_Hack/data/bank_transactions_features.csv"
pd.set_option("display.width", 140)

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded feature-enriched dataset: {df.shape[0]} customers\n")

    q1, q3 = df["Total_Txn_Value"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    print(f"IQR outlier rule on Total_Txn_Value: Q1=Rs.{q1:,.0f}, Q3=Rs.{q3:,.0f}, "
          f"upper fence (Q3 + 1.5*IQR) = Rs.{upper_fence:,.0f}")

    outliers = df[df["High_Value_Outlier"]]
    print(f"\nFlagged: {len(outliers)} / {len(df)} customers ({len(outliers)/len(df)*100:.1f}%)")

    print("\n=== Outlier summary statistics ===")
    print(outliers[["Deposits", "Withdrawals", "Card_Spend", "Total_Txn_Value"]].describe().round(0))

    print(f"\nBase-wide average Total_Txn_Value: Rs.{df['Total_Txn_Value'].mean():,.0f}")
    print(f"Outlier average Total_Txn_Value  : Rs.{outliers['Total_Txn_Value'].mean():,.0f} "
          f"({outliers['Total_Txn_Value'].mean() / df['Total_Txn_Value'].mean():.1f}x base average)")

    print("\n=== Outlier share by Occupation ===")
    occ_summary = df.groupby("Occupation")["High_Value_Outlier"].agg(["sum", "mean"])
    occ_summary.columns = ["Flagged_Count", "Flagged_Share"]
    occ_summary["Flagged_Share"] = (occ_summary["Flagged_Share"] * 100).round(1)
    print(occ_summary.sort_values("Flagged_Count", ascending=False))

    print("\n=== Top 10 highest-value customers (investigation priority) ===")
    top10 = outliers.sort_values("Total_Txn_Value", ascending=False).head(10)
    print(top10[["Customer_ID", "Age_Group", "Occupation", "City", "Total_Txn_Value"]].to_string(index=False))

    print("\n=== Conclusion ===")
    print("Business Owners are over-represented among flagged outliers relative to their population share, "
          "consistent with legitimate high commercial cash-flow rather than fraud per se -- but individual "
          "Salaried/Homemaker accounts among the top-10 list above warrant a closer AML/monitoring review.")
