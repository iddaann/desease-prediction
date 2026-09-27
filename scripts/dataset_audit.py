"""Audit dataset HealthPredict sebelum eksperimen model.

Usage:
    python scripts/dataset_audit.py
    python scripts/dataset_audit.py data/Training.csv data/Testing.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"


def clean(df):
    return df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")


def read_dataset(path):
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    return path, clean(pd.read_csv(path))


def summarize(path, df):
    if "prognosis" not in df.columns:
        raise ValueError(f"{path.name} tidak memiliki kolom 'prognosis'.")

    feature_cols = [c for c in df.columns if c != "prognosis"]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    class_counts = df["prognosis"].dropna().astype(str).value_counts()

    feature_unique = {
        col: sorted(map(str, df[col].dropna().unique().tolist()))[:20]
        for col in feature_cols
    }
    non_binary = [
        col for col in feature_cols
        if set(numeric[col].dropna().unique()).difference({0, 1})
    ]

    row_vectors = numeric.fillna(0).astype(str).agg("|".join, axis=1)
    return {
        "file": str(path.relative_to(BASE_DIR)) if path.is_relative_to(BASE_DIR) else str(path),
        "rows": int(len(df)),
        "features": int(len(feature_cols)),
        "classes": int(class_counts.size),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "target_missing": int(df["prognosis"].isna().sum()),
        "average_symptoms_per_row": round(float(numeric.fillna(0).sum(axis=1).mean()), 4),
        "min_symptoms_per_row": int(numeric.fillna(0).sum(axis=1).min()),
        "max_symptoms_per_row": int(numeric.fillna(0).sum(axis=1).max()),
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "class_min": int(class_counts.min()) if len(class_counts) else 0,
        "class_max": int(class_counts.max()) if len(class_counts) else 0,
        "class_imbalance_ratio": round(float(class_counts.max() / class_counts.min()), 4) if len(class_counts) and class_counts.min() else None,
        "non_binary_features": non_binary,
        "feature_names": feature_cols,
        "feature_unique_values_sample": feature_unique,
        "_row_vectors": row_vectors.tolist(),
    }


def compare(train_report, test_report):
    train_classes = set(train_report["class_distribution"])
    test_classes = set(test_report["class_distribution"])
    overlap = set(train_report["_row_vectors"]).intersection(test_report["_row_vectors"])
    return {
        "classes_only_in_training": sorted(train_classes - test_classes),
        "classes_only_in_testing": sorted(test_classes - train_classes),
        "classes_in_both": len(train_classes & test_classes),
        "exact_feature_vector_overlap": len(overlap),
        "warning": (
            "Exact feature-vector overlap between train and test can make external evaluation optimistic. "
            "This check does not detect near-duplicates."
            if overlap
            else "No exact feature-vector overlap detected by this check."
        ),
    }


def main():
    train_arg = sys.argv[1] if len(sys.argv) > 1 else "data/Training.csv"
    test_arg = sys.argv[2] if len(sys.argv) > 2 else "data/Testing.csv"

    train_path, train_df = read_dataset(train_arg)
    test_path, test_df = read_dataset(test_arg)

    train_report = summarize(train_path, train_df)
    test_report = summarize(test_path, test_df)
    comparison = compare(train_report, test_report)

    train_report.pop("_row_vectors", None)
    test_report.pop("_row_vectors", None)

    report = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "purpose": "Educational dataset audit; not clinical validation.",
        "training": train_report,
        "testing": test_report,
        "train_test_comparison": comparison,
        "limitations": [
            "A public symptom-disease dataset may be synthetic, simplified, or otherwise unlike clinical data.",
            "High test metrics on this dataset do not establish real-world clinical accuracy.",
            "This audit checks structural properties; it does not establish medical validity.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "dataset_audit.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== HealthPredict Dataset Audit ===")
    for name in ("training", "testing"):
        item = report[name]
        print(f"{name}: {item['rows']} rows | {item['features']} features | {item['classes']} classes")
        print(f"  duplicates={item['duplicate_rows']} missing={item['missing_cells']}")
        print(f"  symptoms/row={item['average_symptoms_per_row']}")
        print(f"  class min/max={item['class_min']}/{item['class_max']}")
    print("train/test exact feature-vector overlap:", comparison["exact_feature_vector_overlap"])
    print("report:", output)


if __name__ == "__main__":
    main()
