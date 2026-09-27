"""Evaluate model robustness when symptom information is incomplete.

Usage:
    python scripts/robustness_test.py

The experiment trains a fresh model only on the training split, then evaluates
complete and synthetically incomplete symptom inputs on an unseen holdout.
It is NOT clinical validation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "Training.csv"
REPORT_FILE = BASE_DIR / "reports" / "robustness_report.json"


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.drop(
        columns=[c for c in df.columns if c.lower().startswith("unnamed")],
        errors="ignore",
    )
    df = df.dropna(subset=["prognosis"]).drop_duplicates().reset_index(drop=True)

    features = [c for c in df.columns if c != "prognosis"]
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)

    # Keep one representative row per unique symptom pattern + target.
    unique = pd.concat([X, df["prognosis"].astype(str).rename("prognosis")], axis=1)
    unique = unique.drop_duplicates().reset_index(drop=True)

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


def evaluate(model, X_test, y_test, condition, removed_symptoms):
    pred = model.predict(X_test)
    return {
        "condition": condition,
        "removed_symptoms": removed_symptoms,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }


def main():
    X, y_text, features = load_data()

    # Encode labels independently for this evaluation. The saved production
    # label encoder is not used because this test trains its own fresh model.
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # IMPORTANT: fit only on X_train. X_test remains completely unseen.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    results = [
        evaluate(
            model,
            X_test,
            y_test,
            "complete_symptoms",
            0,
        )
    ]

    for remove_count in (1, 2, 3):
        masked = make_masked_data(
            X_test,
            remove_count,
            seed=42 + remove_count,
        )
        results.append(
            evaluate(
                model,
                masked,
                y_test,
                f"minus_{remove_count}_symptoms",
                remove_count,
            )
        )

    report = {
        "experiment": "incomplete_symptom_robustness",
        "purpose": (
            "Learning-oriented robustness experiment for chatbot-style "
            "incomplete symptom input using a strict unseen holdout."
        ),
        "clinical_validation": False,
        "dataset": {
            "source": "data/Training.csv",
            "rows_after_deduplication": int(len(X)),
            "unique_symptom_patterns_used": int(len(X)),
            "train_rows": int(len(X_train)),
            "holdout_rows": int(len(X_test)),
        },
        "model": {
            "algorithm": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": 42,
            "class_weight": "balanced_subsample",
            "fit_scope": "train_split_only",
        },
        "method": {
            "holdout_random_state": 42,
            "masking": (
                "Randomly remove positive symptom features from each unseen "
                "holdout row."
            ),
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
    REPORT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=== HealthPredict Robustness Test ===")
    print(f"Unique patterns: {len(X)}")
    print(f"Train samples: {len(X_train)}")
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
