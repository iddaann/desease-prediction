"""Evaluate a trained model on the separate Testing.csv file."""
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from preprocessing import load_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def evaluate():
    _, _, X_test, y_test, feature_cols = load_dataset()
    model = joblib.load(MODELS_DIR / "desease_model.joblib")
    encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    y_test_enc = encoder.transform(y_test)
    y_pred_enc = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test_enc, y_pred_enc)
    macro_f1 = f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test_enc, y_pred_enc, average="weighted", zero_division=0)
    top3 = top_k_accuracy_score(
        y_test_enc, y_proba, k=3, labels=np.arange(len(encoder.classes_))
    )

    print(f"Jumlah sampel testing: {len(y_test_enc)}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")
    print(f"Top-3 score : {top3:.4f}")
    print("\nClassification report per kelas:")
    print(classification_report(
        y_test_enc,
        y_pred_enc,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    ))

    matrix = confusion_matrix(
        y_test_enc, y_pred_enc, labels=np.arange(len(encoder.classes_))
    )
    print("Confusion matrix (urutan label mengikuti encoder.classes_):")
    print(matrix)
    print(
        "\nPERINGATAN: skor pada dataset publik yang kecil/terstruktur bukan "
        "estimasi performa klinis. Validasi eksternal tetap dibutuhkan."
    )
    return accuracy


if __name__ == "__main__":
    evaluate()
