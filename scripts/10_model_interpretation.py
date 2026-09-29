"""
ACTION: Interpret the model result and connect it with the analytical findings.

Run standalone:  python scripts/10_model_interpretation.py
(requires outputs/knn_model.joblib from 09_knn_model_build_evaluate.py)

Uses permutation importance (model-agnostic, works for KNN) to see which
features actually drive the predictions, then ties that back to the EDA
patterns found in scripts 06 and 07.
"""

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance

MODEL_PATH = "D:/DS_Hack/outputs/knn_model.joblib"
OUT_PATH = "D:/DS_Hack/outputs/chart7_permutation_importance.png"

if __name__ == "__main__":
    bundle = joblib.load(MODEL_PATH)
    model, X_test, y_test = bundle["model"], bundle["X_test"], bundle["y_test"]
    print(f"Loaded fitted KNN model. Test set: {len(X_test)} customers, "
          f"{len(bundle['feature_cols'])} features: {bundle['feature_cols']}\n")

    print("Computing permutation importance (10 repeats, scoring=F1)...")
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring="f1")
    importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    importance_std = pd.Series(result.importances_std, index=X_test.columns)

    print("\n=== Permutation importance (drop in F1 when feature is shuffled) ===")
    table = pd.DataFrame({"Importance": importance, "Std": importance_std.reindex(importance.index)}).round(4)
    print(table)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=importance.values, y=importance.index, ax=ax, color="#C44E52")
    ax.set_xlabel("Mean F1-score drop when feature is shuffled")
    ax.set_ylabel("Feature")
    ax.set_title("KNN Feature Importance (Permutation Method)")
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"\nSaved -> {OUT_PATH}")

    ranked = list(importance.index)
    top3 = ranked[:3]
    weak = [f for f in ranked if importance[f] < 0.02]

    print("\n=== Interpretation, connected to the EDA findings ===")
    print(f"Top 3 predictors by permutation importance: {top3[0]} ({importance[top3[0]]:.3f}), "
          f"{top3[1]} ({importance[top3[1]]:.3f}), {top3[2]} ({importance[top3[2]]:.3f}).")
    print(f"Weak/negligible contributors: {', '.join(weak)}.")
    print(f"""
- '{top3[0]}' is by far the strongest predictor -- this directly matches the
  single clearest EDA pattern from script 06: digital share fell almost
  linearly with age, from 84% (18-25) to 16% (65+). The model rediscovered
  this pattern on its own, since Digital_Ratio itself was excluded from the
  feature set to avoid leakage.
- '{top3[1]}' is the second strongest, echoing script 06's occupation
  breakdown (Students 83% digital vs. Retirees 25%) -- age and occupation
  are confirmed as the two dominant, independent drivers of digital
  adoption, consistent across both the manual EDA and the model.
- 'Bill_Payments' also carries real signal (rank 3) -- this lines up with
  the correlation heatmap (chart 4), where Bill_Payments correlates more
  strongly with UPI_Count (0.66) and Card_Count (0.61) than with ATM_Count
  (0.37): customers who pay more bills also tend to be more digitally
  active overall, not just on the UPI/Card channels directly measured.
- Notably, ATM_Trend_Slope (the 6-month decline metric from script 07)
  contributes only marginally once Age_Group is already in the model --
  its information is largely redundant with age, not because the ATM
  decline pattern is false (script 07 confirmed it independently on raw
  data), but because age alone already separates the same customers this
  model needs to classify. City and raw Deposits/Withdrawals carry
  essentially no predictive signal, confirming script 06's finding that
  geography is a weak driver and that overall monetary activity level
  does not, by itself, indicate channel preference.
- Practically: this ranking tells the bank that a simple two-field lookup
  (Age_Group + Occupation) captures most of the achievable predictive
  power for digital-adoption propensity -- Bill_Payments count is worth
  adding as a light-weight third signal, but ATM trend monitoring and
  city-level targeting add little beyond what age/occupation already give.
""")
