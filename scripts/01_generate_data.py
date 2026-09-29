"""
Generate a synthetic 6-month customer transaction dataset for the
'Understand Customer Transaction Behaviour' banking case study.

Why synthetic: no suitable ready-made Kaggle dataset matches this exact
schema (per-customer 6-month aggregates of UPI/Card/ATM/Deposits/
Withdrawals/Bill_Payments alongside Age_Group/Occupation/City). Patterns
useful for the assignment (digital adoption by segment, declining ATM
usage in some segments, outlier transactors) are seeded deliberately so
the EDA and KNN model have real signal to recover -- exactly like a
realistic bank extract would.

A few intentional data-quality issues (missing values, duplicate rows,
inconsistent text casing, stray whitespace, a handful of negative
amounts from refund/reversal entries) are injected so the cleaning step
in the next script has real work to do.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_CUSTOMERS = 1500

AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
AGE_WEIGHTS = [0.16, 0.26, 0.22, 0.18, 0.12, 0.06]

OCCUPATIONS = ["Student", "Salaried", "Self-Employed", "Business Owner", "Retired", "Homemaker"]

# Occupation distribution conditioned on age group (rough real-world plausibility)
OCC_BY_AGE = {
    "18-25":  {"Student": 0.55, "Salaried": 0.30, "Self-Employed": 0.05, "Business Owner": 0.03, "Retired": 0.00, "Homemaker": 0.07},
    "26-35":  {"Student": 0.05, "Salaried": 0.55, "Self-Employed": 0.15, "Business Owner": 0.10, "Retired": 0.00, "Homemaker": 0.15},
    "36-45":  {"Student": 0.00, "Salaried": 0.50, "Self-Employed": 0.18, "Business Owner": 0.15, "Retired": 0.01, "Homemaker": 0.16},
    "46-55":  {"Student": 0.00, "Salaried": 0.42, "Self-Employed": 0.18, "Business Owner": 0.18, "Retired": 0.05, "Homemaker": 0.17},
    "56-65":  {"Student": 0.00, "Salaried": 0.20, "Self-Employed": 0.12, "Business Owner": 0.13, "Retired": 0.40, "Homemaker": 0.15},
    "65+":    {"Student": 0.00, "Salaried": 0.03, "Self-Employed": 0.05, "Business Owner": 0.07, "Retired": 0.75, "Homemaker": 0.10},
}

TIER1_CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune"]
TIER2_CITIES = ["Jaipur", "Lucknow", "Nagpur", "Indore", "Bhopal", "Coimbatore"]
ALL_CITIES = TIER1_CITIES + TIER2_CITIES
CITY_WEIGHTS = [0.11, 0.11, 0.10, 0.09, 0.09, 0.08] + [0.07, 0.07, 0.06, 0.07, 0.08, 0.07]

# Baseline digital-transaction propensity by age group (drives UPI/Card vs ATM mix)
DIGITAL_BASE = {"18-25": 0.85, "26-35": 0.80, "36-45": 0.65, "46-55": 0.45, "56-65": 0.30, "65+": 0.15}

# Monthly ATM-usage trend slope by age group (negative = declining over the 6 months)
ATM_TREND_SLOPE = {"18-25": -0.09, "26-35": -0.07, "36-45": -0.03, "46-55": -0.01, "56-65": 0.01, "65+": 0.02}

# Occupation income/activity multiplier (affects overall transaction volume & value)
OCC_ACTIVITY = {"Student": 0.45, "Salaried": 1.0, "Self-Employed": 1.15, "Business Owner": 1.5, "Retired": 0.7, "Homemaker": 0.6}

CITY_TIER_MULT = {c: 1.25 for c in TIER1_CITIES}
CITY_TIER_MULT.update({c: 0.95 for c in TIER2_CITIES})


def sample_occupation(age_group):
    probs = OCC_BY_AGE[age_group]
    occs = list(probs.keys())
    weights = list(probs.values())
    return RNG.choice(occs, p=weights)


def generate():
    age_groups = RNG.choice(AGE_GROUPS, size=N_CUSTOMERS, p=AGE_WEIGHTS)
    occupations = np.array([sample_occupation(ag) for ag in age_groups])
    cities = RNG.choice(ALL_CITIES, size=N_CUSTOMERS, p=CITY_WEIGHTS)

    rows = []
    for i in range(N_CUSTOMERS):
        ag = age_groups[i]
        occ = occupations[i]
        city = cities[i]

        activity = OCC_ACTIVITY[occ] * CITY_TIER_MULT[city] * RNG.lognormal(mean=0, sigma=0.35)
        digital_p = np.clip(DIGITAL_BASE[ag] + RNG.normal(0, 0.08), 0.05, 0.97)

        # total "channel" transactions split between digital (UPI+Card) and ATM
        total_channel_txns = max(5, RNG.poisson(60 * activity))
        digital_txns = RNG.binomial(total_channel_txns, digital_p)
        atm_txns = total_channel_txns - digital_txns

        upi_share = np.clip(RNG.normal(0.65, 0.1), 0.3, 0.9)
        upi_count = int(round(digital_txns * upi_share))
        card_count = max(0, digital_txns - upi_count)

        # monthly ATM breakdown with segment-specific trend + noise, non-negative
        base_month_atm = max(atm_txns / 6, 0.5)
        slope = ATM_TREND_SLOPE[ag]
        monthly_atm = []
        for m in range(1, 7):
            val = base_month_atm * (1 + slope * (m - 3.5))
            val = max(0, RNG.poisson(max(val, 0.1)))
            monthly_atm.append(val)
        atm_count = int(sum(monthly_atm))

        card_spend = round(card_count * RNG.lognormal(mean=6.3, sigma=0.5), 2) if card_count > 0 else 0.0

        deposits = round(RNG.lognormal(mean=10.2, sigma=0.55) * activity, 2)
        withdrawals = round(deposits * RNG.uniform(0.4, 0.9) * RNG.lognormal(0, 0.15), 2)
        bill_payments = max(0, RNG.poisson(6 * activity * (0.6 + 0.4 * digital_p)))

        rows.append({
            "Customer_ID": f"CUST{i+1:05d}",
            "Age_Group": ag,
            "Occupation": occ,
            "City": city,
            "UPI_Count": upi_count,
            "Card_Count": card_count,
            "Card_Spend": card_spend,
            "ATM_Count": atm_count,
            "ATM_M1": monthly_atm[0], "ATM_M2": monthly_atm[1], "ATM_M3": monthly_atm[2],
            "ATM_M4": monthly_atm[3], "ATM_M5": monthly_atm[4], "ATM_M6": monthly_atm[5],
            "Deposits": deposits,
            "Withdrawals": withdrawals,
            "Bill_Payments": bill_payments,
        })

    df = pd.DataFrame(rows)

    # ---- inject a small number of unusually high / investigation-worthy transactors ----
    outlier_idx = RNG.choice(df.index, size=int(0.03 * N_CUSTOMERS), replace=False)
    for idx in outlier_idx:
        spike = RNG.uniform(6, 15)
        df.loc[idx, "Deposits"] = round(df.loc[idx, "Deposits"] * spike, 2)
        df.loc[idx, "Withdrawals"] = round(df.loc[idx, "Withdrawals"] * spike * RNG.uniform(0.8, 1.2), 2)

    # ================= inject data-quality issues for the cleaning step =================
    df_dirty = df.copy()

    # 1. Missing values scattered across a few columns
    for col, frac in [("Occupation", 0.02), ("City", 0.015), ("Bill_Payments", 0.01), ("Card_Spend", 0.01)]:
        idx = RNG.choice(df_dirty.index, size=int(frac * N_CUSTOMERS), replace=False)
        df_dirty.loc[idx, col] = np.nan

    # 2. Inconsistent text casing / stray whitespace
    idx = RNG.choice(df_dirty.index, size=40, replace=False)
    df_dirty.loc[idx, "City"] = df_dirty.loc[idx, "City"].astype(str).str.upper()
    idx = RNG.choice(df_dirty.index, size=30, replace=False)
    df_dirty.loc[idx, "Occupation"] = df_dirty.loc[idx, "Occupation"].apply(
        lambda x: f"  {x.lower()} " if isinstance(x, str) else x
    )

    # 3. A few negative amounts (refund/reversal artefacts that need correction)
    idx = RNG.choice(df_dirty.index, size=8, replace=False)
    df_dirty.loc[idx, "Withdrawals"] = -df_dirty.loc[idx, "Withdrawals"].abs() * 0.1

    # 4. Duplicate rows (system export duplication)
    dupes = df_dirty.sample(n=12, random_state=7)
    df_dirty = pd.concat([df_dirty, dupes], ignore_index=True)

    # 5. Shuffle row order
    df_dirty = df_dirty.sample(frac=1, random_state=1).reset_index(drop=True)

    return df_dirty


if __name__ == "__main__":
    data = generate()
    out_path = "D:/DS_Hack/data/bank_transactions_raw.csv"
    data.to_csv(out_path, index=False)
    print(f"Generated {len(data)} rows -> {out_path}")
    print(data.head())
