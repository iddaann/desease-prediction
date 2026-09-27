"""Evaluate model robustness when symptom information is incomplete.

Usage:
    python scripts/robustness_test.py

The experiment is intentionally a learning-oriented robustness test. It removes
1, 2, and 3 positive symptoms from each unique symptom pattern and measures
how predictions change. It is NOT clinical validation.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "Training.csv"
MODEL_FILE = BASE_DIR / "models" / "disease" / "model.joblib"
ENCODER_FILE = BASE_DIR / "models" / "disease" / "label_encoder.joblib"
FEATURES_FILE = BASE_DIR / "models" / "disease" / "feature_columns.joblib"
REPORT_FILE = BASE_DIR / "reports" / "robustness_report.json"


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    df = df.dropna(subset=["prognosis"]).drop_duplicates().reset_index(drop=True)

    features = [c for c in df.columns if c != "prognosis"]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["prognosis"].astype(str)

    # Keep one representative row per symptom pattern + target.
    unique = pd.concat([X, y.rename("prognosis")], axis=1).drop_duplicates().reset_index(drop=True)
    return unique[features], unique["prognosis"], features


def make_masked_data(X, remove_count, seed=42):
    rng = np.random.default_rng(seed)
    masked = X.copy()

    for i in range(len(masked)):
        positive = np.flatnonzero(masked.iloc[i].to_numpy() > 0)
        if len(positive) == 0:
            continue
        count = min(remove_count, len(positive))
        selected = rng.choice(positive, size=count, replace=False)
        masked.iloc[i, selected] = 0

    return masked


def main():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model belum ditemukan. Jalankan: python -m src.train"
        )

    X, y_text, features = load_data()
    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    saved_features = joblib.load(FEATURES_FILE)

    if list(saved_features) != list(features):
        raise ValueError("Feature columns dataset tidak sama dengan model.")

    y = encoder.transform(y_text)

    # Evaluate only on a holdout of unique symptom patterns.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # The saved model is trained on the full deduplicated training data.
    # This experiment intentionally measures input robustness of that model.
    results = []

    original_pred = model.predict(X_test)
    results.append(
        {
            "condition": "complete_symptoms",
            "removed_symptoms": 0,
            "accuracy": float(accuracy_score(y_test, original_pred)),
            "macro_f1": float(f1_score(y_test, original_pred, average="macro")),
        }
    )

    for remove_count in (1, 2, 3):
        masked = make_masked_data(X_test, remove_count, seed=42 + remove_count)
        pred = model.predict(masked)
        results.append(
            {
                "condition": f"minus_{remove_count}_symptoms",
                "removed_symptoms": remove_count,
                "accuracy": float(accuracy_score(y_test, pred)),
                "macro_f1": float(f1_score(y_test, pred, average="macro")),
            }
        )

    report = {
        "experiment": "incomplete_symptom_robustness",
        "purpose": "Learning-oriented robustness experiment for chatbot-style incomplete symptom input.",
        "clinical_validation": False,
        "dataset": {
            "source": "data/Training.csv",
            "rows_after_deduplication": int(len(X)),
            "unique_symptom_patterns_used": int(len(X)),
            "holdout_rows": int(len(X_test)),
        },
        "model": {
            "artifact": str(MODEL_FILE.relative_to(BASE_DIR)),
            "version": (
                (BASE_DIR / "models" / "disease" / "version.txt").read_text().strip()
                if (BASE_DIR / "models" / "disease" / "version.txt").exists()
                else None
            ),
        },
        "method": {
            "holdout_random_state": 42,
            "masking": "Randomly remove positive symptom features from each holdout row.",
            "levels": [1, 2, 3],
        },
        "results": results,
        "limitations": [
            "The experiment uses the same source dataset as training.",
            "It is not an independent external validation.",
            "Removing symptoms synthetically does not reproduce real patient reporting.",
            "Results must not be interpreted as clinical diagnostic performance.",
        ],
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== HealthPredict Robustness Test ===")
    print(f"Unique patterns: {len(X)}")
    print(f"Holdout samples: {len(X_test)}")
    for item in results:
        print(
            f"{item['condition']}: "
            f"accuracy={item['accuracy']:.4f} "
            f"macro_f1={item['macro_f1']:.4f}"
        )
    print(f"report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
