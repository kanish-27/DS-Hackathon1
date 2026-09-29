"""
ACTION: Build and evaluate a KNN model for the prediction target defined in the challenge.

Run standalone:  python scripts/09_knn_model_build_evaluate.py
(requires data/bank_transactions_features.csv from 03_eda_descriptive_stats.py)

Target: Digital_Preferring (1 = customer's UPI+Card transaction count exceeds
the base median; 0 = otherwise).

Features deliberately EXCLUDE UPI_Count/Card_Count/Digital_Ratio (which define
the label) to avoid leakage: Age_Group, Occupation, City, ATM_Count,
ATM_Trend_Slope, Deposits, Withdrawals, Bill_Payments -- signals the bank
always has on file, even for customers with thin digital transaction history.

Saves the fitted model to outputs/knn_model.joblib for reuse by
10_model_interpretation.py.
"""

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)

FEATURES_PATH = "D:/DS_Hack/data/bank_transactions_features.csv"
MODEL_PATH = "D:/DS_Hack/outputs/knn_model.joblib"
TEST_SPLIT_PATH = "D:/DS_Hack/outputs/knn_test_split.csv"
AGE_ORDER = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]

FEATURE_COLS_NUM = ["ATM_Count", "ATM_Trend_Slope", "Deposits", "Withdrawals", "Bill_Payments"]
FEATURE_COLS_AGE = ["Age_Group"]
FEATURE_COLS_NOMINAL = ["Occupation", "City"]

if __name__ == "__main__":
    df = pd.read_csv(FEATURES_PATH)
    print(f"Loaded feature-enriched dataset: {df.shape[0]} customers\n")

    digital_count = df["UPI_Count"] + df["Card_Count"]
    threshold = digital_count.median()
    df["Digital_Preferring"] = (digital_count > threshold).astype(int)
    print(f"Target definition: Digital_Preferring = 1 if (UPI_Count + Card_Count) > {threshold} else 0")
    print("Target class balance:")
    print(df["Digital_Preferring"].value_counts(normalize=True).round(3))

    X = df[FEATURE_COLS_NUM + FEATURE_COLS_AGE + FEATURE_COLS_NOMINAL]
    y = df["Digital_Preferring"]
    print(f"\nFeature columns used ({len(X.columns)}): {list(X.columns)}")
    print("(UPI_Count / Card_Count / Digital_Ratio excluded -- they define the label)")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nTrain/test split: {len(X_train)} train, {len(X_test)} test (stratified, 75/25)")

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), FEATURE_COLS_NUM),
        ("age", OrdinalEncoder(categories=[AGE_ORDER]), FEATURE_COLS_AGE),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS_NOMINAL),
    ])
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("knn", KNeighborsClassifier())])

    param_grid = {
        "knn__n_neighbors": list(range(3, 31, 2)),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    }
    print("\nRunning 5-fold GridSearchCV over k=3..29, weights=[uniform,distance], metric=[euclidean,manhattan]...")
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("\n=== Best model ===")
    print("Best hyperparameters:", grid.best_params_)
    print("Best 5-fold CV F1   :", round(grid.best_score_, 4))

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    print("\n=== Test-set evaluation (held-out 25%, n={}) ===".format(len(y_test)))
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    print("\n" + classification_report(y_test, y_pred, target_names=["Traditional", "Digital-Preferring"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(pd.DataFrame(cm, index=["Actual: Traditional", "Actual: Digital-Preferring"],
                        columns=["Pred: Traditional", "Pred: Digital-Preferring"]))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Traditional", "Digital-Preferring"],
                yticklabels=["Traditional", "Digital-Preferring"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"KNN Confusion Matrix (k={grid.best_params_['knn__n_neighbors']})\n"
                 f"Accuracy={metrics['accuracy']:.1%}  F1={metrics['f1']:.2f}")
    fig.tight_layout()
    cm_path = "D:/DS_Hack/outputs/chart5_knn_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix chart -> {cm_path}")

    joblib.dump({"model": best_model, "X_test": X_test, "y_test": y_test,
                 "feature_cols": list(X.columns)}, MODEL_PATH)
    print(f"\nSaved fitted model + test split -> {MODEL_PATH}")
