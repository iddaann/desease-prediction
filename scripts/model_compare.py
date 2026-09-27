"""Compare baseline classifiers on the same train/validation split.

Usage:
    python scripts/model_compare.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"


def load_data():
    df = pd.read_csv(BASE_DIR / "data" / "Training.csv")
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    df = df.dropna(subset=["prognosis"]).drop_duplicates().reset_index(drop=True)
    features = [c for c in df.columns if c != "prognosis"]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df["prognosis"].astype(str))
    return X, y


def evaluate(name, model, X_train, X_test, y_train, y_test, cv):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    return {
        "model": name,
        "holdout_accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "holdout_precision_macro": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
        "holdout_recall_macro": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
        "holdout_f1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
        "cv_f1_macro_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_macro_std": round(float(cv_f1.std()), 4),
    }


def main():
    X, y = load_data()
    min_class = int(pd.Series(y).value_counts().min())
    if min_class < 2:
        raise ValueError("Setiap kelas harus memiliki minimal dua contoh.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(0.2, 2 / len(X)), random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=min(5, min_class), shuffle=True, random_state=42)

    models = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("SVM RBF", SVC(class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )),
    ]

    results = [evaluate(name, model, X_train, X_test, y_train, y_test, cv) for name, model in models]
    payload = {
        "purpose": "Educational model comparison; metrics do not imply clinical validity.",
        "results": results,
        "selection_note": "Do not select a model from accuracy alone; inspect macro F1, recall, confusion matrix, and dataset limitations.",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "model_comparison.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== HealthPredict Model Comparison ===")
    for item in results:
        print(
            f"{item['model']}: accuracy={item['holdout_accuracy']} "
            f"macro_f1={item['holdout_f1_macro']} "
            f"cv_macro_f1={item['cv_f1_macro_mean']}±{item['cv_f1_macro_std']}"
        )
    print("report:", output)


if __name__ == "__main__":
    main()
