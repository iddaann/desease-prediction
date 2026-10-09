"""Evaluasi silang dengan grouping berdasarkan kombinasi gejala.

Peringatan: dataset publik ini bersifat sintetis/terstruktur dan hasilnya
bukan estimasi performa klinis pada pasien nyata.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def evaluate_cv():
    df = pd.read_csv(DATA_DIR / "Training.csv")
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
    if "prognosis" not in df:
        raise ValueError("Kolom target 'prognosis' tidak ditemukan.")

    print(f"Baris mentah: {len(df)}")
    exact_duplicates = int(df.duplicated().sum())
    print(f"Duplikat baris persis: {exact_duplicates}")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Baris setelah deduplikasi: {len(df)}")
    print(f"Jumlah penyakit: {df['prognosis'].nunique()}")
    print(f"Jumlah fitur: {df.shape[1] - 1}")

    X = df.drop(columns="prognosis").astype(int)
    y = df["prognosis"].astype(str)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Kelompokkan baris yang memiliki kombinasi fitur/gejala identik.
    # Satu kombinasi tidak boleh muncul sekaligus di train dan validation.
    groups = pd.util.hash_pandas_object(X, index=False).astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    metrics = []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(X, y_encoded, groups=groups), start=1
    ):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        model = RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        proba = model.predict_proba(X_val)
        labels = model.classes_
        accuracy = accuracy_score(y_val, pred)
        macro_f1 = f1_score(y_val, pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_val, pred, average="weighted", zero_division=0)
        # Model tiap fold mungkin tidak memuat semua kelas; top-k dihitung
        # hanya atas kelas yang benar-benar dipelajari fold tersebut.
        top3 = top_k_accuracy_score(y_val, proba, k=min(3, len(labels)), labels=labels)
        metrics.append((accuracy, macro_f1, weighted_f1, top3))
        print(
            f"Fold {fold}: accuracy={accuracy:.4f}, macro-F1={macro_f1:.4f}, "
            f"weighted-F1={weighted_f1:.4f}, top-3={top3:.4f}"
        )
        print(classification_report(
            y_val, pred,
            labels=np.unique(y_val),
            target_names=encoder.inverse_transform(np.unique(y_val)),
            zero_division=0,
        ))

    arr = np.asarray(metrics)
    print("\n=== RINGKASAN 5-FOLD (rata-rata ± standar deviasi) ===")
    for idx, name in enumerate(("Accuracy", "Macro F1", "Weighted F1", "Top-3 accuracy")):
        print(f"{name}: {arr[:, idx].mean():.4f} ± {arr[:, idx].std():.4f}")
    print(
        "\nCATATAN: evaluasi berbasis dataset ini tidak membuktikan akurasi klinis. "
        "Validasi eksternal dengan data yang representatif dan ditinjau tenaga "
        "kesehatan tetap diperlukan."
    )


if __name__ == "__main__":
    evaluate_cv()
