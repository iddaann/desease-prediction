from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    top_k_accuracy_score,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def evaluate_cv():
    print("Memuat dataset...")

    df = pd.read_csv(DATA_DIR / "Training.csv")

    # Bersihkan kolom yang tidak diperlukan
    junk_cols = [
        c for c in df.columns
        if c.lower().startswith("unnamed")
    ]

    if junk_cols:
        df = df.drop(columns=junk_cols)

    print(f"Data awal      : {df.shape[0]} baris")
    
    # Hapus duplicate hanya dari memory
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"Data setelah dedup: {df.shape[0]} baris")
    print(f"Jumlah penyakit : {df['prognosis'].nunique()}")
    print(f"Jumlah fitur    : {df.shape[1] - 1}")
    print()

    X = df.drop(columns="prognosis")
    y = df["prognosis"]

    # Encode label penyakit
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    accuracies = []
    macro_f1s = []
    weighted_f1s = []
    top3_scores = []
    top5_scores = []

    print("=== 5-FOLD CROSS VALIDATION ===")

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(X, y_encoded), start=1
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y_encoded[train_idx]
        y_val = y_encoded[val_idx]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # Prediksi kelas
        y_pred = model.predict(X_val)

        # Probabilitas setiap kelas
        y_proba = model.predict_proba(X_val)

        accuracy = accuracy_score(y_val, y_pred)

        macro_f1 = f1_score(
            y_val,
            y_pred,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            y_val,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        # top_k_accuracy membutuhkan seluruh kelas
        top3 = top_k_accuracy_score(
            y_val,
            y_proba,
            k=3,
            labels=np.arange(len(encoder.classes_)),
        )

        top5 = top_k_accuracy_score(
            y_val,
            y_proba,
            k=5,
            labels=np.arange(len(encoder.classes_)),
        )

        accuracies.append(accuracy)
        macro_f1s.append(macro_f1)
        weighted_f1s.append(weighted_f1)
        top3_scores.append(top3)
        top5_scores.append(top5)

        print(
            f"Fold {fold}: "
            f"Accuracy={accuracy:.4f} | "
            f"Macro F1={macro_f1:.4f} | "
            f"Top-3={top3:.4f} | "
            f"Top-5={top5:.4f}"
        )

    print()
    print("=== HASIL AKHIR ===")

    print(
        f"Accuracy : "
        f"{np.mean(accuracies):.4f} "
        f"(±{np.std(accuracies):.4f})"
    )

    print(
        f"Macro F1 : "
        f"{np.mean(macro_f1s):.4f} "
        f"(±{np.std(macro_f1s):.4f})"
    )

    print(
        f"Weighted F1 : "
        f"{np.mean(weighted_f1s):.4f} "
        f"(±{np.std(weighted_f1s):.4f})"
    )

    print(
        f"Top-3 Accuracy : "
        f"{np.mean(top3_scores):.4f} "
        f"(±{np.std(top3_scores):.4f})"
    )

    print(
        f"Top-5 Accuracy : "
        f"{np.mean(top5_scores):.4f} "
        f"(±{np.std(top5_scores):.4f})"
    )


if __name__ == "__main__":
    evaluate_cv()