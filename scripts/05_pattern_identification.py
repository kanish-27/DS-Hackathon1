"""
ACTION: Identify different transaction behaviour patterns.

Run standalone:  python scripts/05_pattern_identification.py
(requires data/bank_transactions_features.csv from 03_eda_descriptive_stats.py)

Segments customers into behaviour profiles using a 2x2 split on
transaction activity level (Total_Channel_Txns) and digital preference
(Digital_Ratio), then profiles each segment.
"""

import pandas as pd

FEATURES_PATH = "D:/DS_Hack/data/bank_transactions_features.csv"
pd.set_option("display.width", 140)


def segment(df: pd.DataFrame) -> pd.DataFrame:
    activity_median = df["Total_Channel_Txns"].median()
    digital_median = df["Digital_Ratio"].median()

    def label(row):
        activity = "High-Activity" if row["Total_Channel_Txns"] >= activity_median else "Low-Activity"
        style = "Digital-Leaning" if row["Digital_Ratio"] >= digital_median else "Traditional-Leaning"
        return f"{style} / {activity}"

    df["Behaviour_Segment"] = df.apply(label, axis=1)
    return df, activity_median, digital_median


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded feature-enriched dataset: {df.shape[0]} customers\n")

    df, activity_median, digital_median = segment(df)
    print(f"Activity split threshold (median total channel transactions): {activity_median:.0f}")
    print(f"Digital split threshold (median digital ratio): {digital_median:.2f}\n")

    print("=== Behaviour segment sizes ===")
    counts = df["Behaviour_Segment"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    summary = pd.DataFrame({"Customers": counts, "Share_%": pct})
    print(summary)

    print("\n=== Segment profiles (average metrics) ===")
    profile = df.groupby("Behaviour_Segment").agg(
        Avg_Digital_Ratio=("Digital_Ratio", "mean"),
        Avg_ATM_Count=("ATM_Count", "mean"),
        Avg_Deposits=("Deposits", "mean"),
        Avg_Total_Txn_Value=("Total_Txn_Value", "mean"),
        Most_Common_Age_Group=("Age_Group", lambda s: s.mode()[0]),
        Most_Common_Occupation=("Occupation", lambda s: s.mode()[0]),
    ).round(2)
    print(profile)

    print("\n=== Narrative summary ===")
    for seg, row in profile.iterrows():
        share = summary.loc[seg, "Share_%"]
        print(f"- {seg} ({share}% of customers, typically {row['Most_Common_Age_Group']}, "
              f"{row['Most_Common_Occupation']}): avg digital ratio {row['Avg_Digital_Ratio']:.0%}, "
              f"avg total 6-month value Rs.{row['Avg_Total_Txn_Value']:,.0f}")
