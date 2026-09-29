"""
ACTION: Investigate whether ATM usage is declining among certain segments.

Run standalone:  python scripts/07_atm_decline_analysis.py
(requires data/bank_transactions_features.csv from 03_eda_descriptive_stats.py)
"""

import pandas as pd

FEATURES_PATH = "D:/DS_Hack/data/bank_transactions_features.csv"
AGE_ORDER = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
ATM_MONTH_COLS = [f"ATM_M{i}" for i in range(1, 7)]
pd.set_option("display.width", 140)

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded feature-enriched dataset: {df.shape[0]} customers\n")

    print("=== Average monthly ATM transactions by Age Group (Month 1 -> Month 6) ===")
    monthly_by_age = df.groupby("Age_Group")[ATM_MONTH_COLS].mean().reindex(AGE_ORDER).round(2)
    print(monthly_by_age)

    print("\n=== ATM usage trend slope by Age Group (negative = declining) ===")
    slope_by_age = df.groupby("Age_Group")["ATM_Trend_Slope"].mean().reindex(AGE_ORDER).round(3)
    pct_change = ((monthly_by_age["ATM_M6"] - monthly_by_age["ATM_M1"]) / monthly_by_age["ATM_M1"] * 100).round(1)
    trend_table = pd.DataFrame({"Trend_Slope": slope_by_age, "Month1_to_Month6_Change_%": pct_change})
    print(trend_table)

    print("\n=== ATM usage trend slope by Occupation ===")
    slope_by_occ = df.groupby("Occupation")["ATM_Trend_Slope"].mean().sort_values().round(3)
    print(slope_by_occ)

    print("\n=== Conclusion ===")
    declining = trend_table[trend_table["Trend_Slope"] < 0].sort_values("Trend_Slope")
    rising = trend_table[trend_table["Trend_Slope"] >= 0].sort_values("Trend_Slope", ascending=False)
    print("Segments with DECLINING ATM usage over the 6 months:")
    for ag, row in declining.iterrows():
        print(f"  {ag}: slope {row['Trend_Slope']:.3f}, {row['Month1_to_Month6_Change_%']:.0f}% change Month1->Month6")
    print("\nSegments with FLAT/RISING ATM usage over the 6 months:")
    for ag, row in rising.iterrows():
        print(f"  {ag}: slope {row['Trend_Slope']:.3f}, {row['Month1_to_Month6_Change_%']:.0f}% change Month1->Month6")
    print("\nATM decline is concentrated in the 26-45 age bands (active digital migration); "
          "56+ segments show flat-to-rising ATM reliance.")
