"""
ACTION: Create clear visualisations for the important patterns in the data.

Run standalone:  python scripts/04_visualizations.py
(requires data/bank_transactions_features.csv from 03_eda_descriptive_stats.py)

Produces the 4 core visualisations for the case study:
  1. Digital transaction preference by age group
  2. Monthly ATM usage trend by age group
  3. Outlier detection - transaction value by occupation
  4. Correlation heatmap of key behaviour metrics
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
FEATURES_PATH = "D:/DS_Hack/data/bank_transactions_features.csv"
OUT_DIR = "D:/DS_Hack/outputs"
AGE_ORDER = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]


def chart1_digital_ratio_by_age(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="Age_Group", y="Digital_Ratio", order=AGE_ORDER, ax=ax, color="#4C72B0")
    ax.set_title("Digital Transaction Preference by Age Group")
    ax.set_ylabel("Avg. Digital Share of Transactions (UPI + Card)")
    ax.set_xlabel("Age Group")
    ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0%}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = f"{OUT_DIR}/chart1_digital_ratio_by_age.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def chart2_atm_trend_by_age(df):
    atm_month_cols = [f"ATM_M{i}" for i in range(1, 7)]
    trend_df = df.groupby("Age_Group")[atm_month_cols].mean().reindex(AGE_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    for ag in AGE_ORDER:
        ax.plot(range(1, 7), trend_df.loc[ag], marker="o", label=ag)
    ax.set_title("Average Monthly ATM Transactions by Age Group (6-Month Trend)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg. ATM Transactions")
    ax.legend(title="Age Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    path = f"{OUT_DIR}/chart2_atm_trend_by_age.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def chart3_outliers_by_occupation(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Occupation", y="Total_Txn_Value", ax=ax, color="#55A868")
    sns.stripplot(data=df[df["High_Value_Outlier"]], x="Occupation", y="Total_Txn_Value",
                   ax=ax, color="red", size=5, alpha=0.7, label="Flagged outlier")
    ax.set_title("Total Transaction Value by Occupation (Outliers Highlighted)")
    ax.set_ylabel("Total 6-Month Transaction Value (Rs.)")
    ax.set_xlabel("Occupation")
    plt.xticks(rotation=20)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:1], labels[:1])
    fig.tight_layout()
    path = f"{OUT_DIR}/chart3_outliers_by_occupation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


def chart4_correlation_heatmap(df):
    numeric_cols = ["UPI_Count", "Card_Count", "ATM_Count", "Deposits", "Withdrawals",
                     "Bill_Payments", "Digital_Ratio", "ATM_Trend_Slope", "Total_Txn_Value"]
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation Between Key Transaction Behaviour Metrics")
    fig.tight_layout()
    path = f"{OUT_DIR}/chart4_correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {path}")


if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded feature-enriched dataset: {df.shape[0]} customers\n")

    print("Generating 4 visualisations...")
    chart1_digital_ratio_by_age(df)
    chart2_atm_trend_by_age(df)
    chart3_outliers_by_occupation(df)
    chart4_correlation_heatmap(df)
    print("\nAll 4 visualisations saved to", OUT_DIR)
