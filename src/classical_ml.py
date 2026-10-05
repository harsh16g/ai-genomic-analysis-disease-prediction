"""
classical_ml.py
-----------------
Module 2: Classical ML classifier comparison (CO4).

Trains Random Forest, SVM, and Logistic Regression on the k-mer feature
matrix and compares them with cross-validation + a held-out test set.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


MODELS = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=2000),
}


def prepare_xy(df: pd.DataFrame):
    X = df.drop(columns=["label"]).values
    y_raw = df["label"].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    return X, y, encoder


def train_and_compare(df: pd.DataFrame, test_size: float = 0.25, cv_folds: int = 5) -> dict:
    """
    Trains every model in MODELS, evaluates on a held-out test split,
    and also reports cross-validated accuracy on the training data.

    Returns a results dict per model with fitted estimator + metrics,
    plus 'label_encoder' and 'scaler' for downstream use (e.g. a CLI
    that predicts on new sequences).
    """
    X, y, encoder = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {"label_encoder": encoder, "scaler": scaler}
    for name, model in MODELS.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv_folds)

        results[name] = {
            "model": model,
            "test_accuracy": round(acc, 4),
            "cv_mean_accuracy": round(cv_scores.mean(), 4),
            "cv_std": round(cv_scores.std(), 4),
            "classification_report": classification_report(
                y_test, preds, target_names=encoder.classes_, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        }

    return results


def print_comparison(results: dict):
    print("\n=== Classical ML Model Comparison ===")
    for name, r in results.items():
        if name in ("label_encoder", "scaler"):
            continue
        print(f"\n--- {name} ---")
        print(f"Test accuracy:      {r['test_accuracy']}")
        print(f"CV accuracy (mean±std): {r['cv_mean_accuracy']} ± {r['cv_std']}")
        print(r["classification_report"])
