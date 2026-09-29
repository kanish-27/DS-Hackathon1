"""
ACTION: Determine which customer groups prefer digital transactions.

Run standalone:  python scripts/06_digital_preference_analysis.py
(requires data/bank_transactions_features.csv from 03_eda_descriptive_stats.py)
"""

import pandas as pd

FEATURES_PATH = "D:/DS_Hack/data/bank_transactions_features.csv"
AGE_ORDER = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
pd.set_option("display.width", 140)

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded feature-enriched dataset: {df.shape[0]} customers\n")

    print("=== Digital transaction share by Age Group ===")
    by_age = df.groupby("Age_Group")["Digital_Ratio"].agg(["mean", "count"]).reindex(AGE_ORDER)
    by_age.columns = ["Avg_Digital_Ratio", "Customers"]
    by_age["Avg_Digital_Ratio"] = by_age["Avg_Digital_Ratio"].round(3)
    print(by_age)

    print("\n=== Digital transaction share by Occupation ===")
    by_occ = df.groupby("Occupation")["Digital_Ratio"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    by_occ.columns = ["Avg_Digital_Ratio", "Customers"]
    by_occ["Avg_Digital_Ratio"] = by_occ["Avg_Digital_Ratio"].round(3)
    print(by_occ)

    print("\n=== Digital transaction share by City ===")
    by_city = df.groupby("City")["Digital_Ratio"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    by_city.columns = ["Avg_Digital_Ratio", "Customers"]
    by_city["Avg_Digital_Ratio"] = by_city["Avg_Digital_Ratio"].round(3)
    print(by_city)
    print(f"\nCity range: {by_city['Avg_Digital_Ratio'].min():.0%} - {by_city['Avg_Digital_Ratio'].max():.0%} "
          f"(spread of {(by_city['Avg_Digital_Ratio'].max() - by_city['Avg_Digital_Ratio'].min()):.0%} -> weak driver vs. age/occupation)")

    print("\n=== Conclusion ===")
    top_age = by_age["Avg_Digital_Ratio"].idxmax()
    bottom_age = by_age["Avg_Digital_Ratio"].idxmin()
    top_occ = by_occ["Avg_Digital_Ratio"].idxmax()
    bottom_occ = by_occ["Avg_Digital_Ratio"].idxmin()
    print(f"Most digital-preferring: {top_age} age group ({by_age.loc[top_age, 'Avg_Digital_Ratio']:.0%}), "
          f"{top_occ} occupation ({by_occ.loc[top_occ, 'Avg_Digital_Ratio']:.0%})")
    print(f"Least digital-preferring: {bottom_age} age group ({by_age.loc[bottom_age, 'Avg_Digital_Ratio']:.0%}), "
          f"{bottom_occ} occupation ({by_occ.loc[bottom_occ, 'Avg_Digital_Ratio']:.0%})")
    print("Age and occupation are the primary drivers of digital preference; city/location is a weak driver.")
