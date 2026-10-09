"""Train Random Forest after removing exact duplicate labeled examples."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from preprocessing import load_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train():
    X_train, y_train, _, _, feature_cols = load_dataset()
    raw_count = len(X_train)

    # Deduplikasi seluruh baris fitur + label, bukan fitur saja; baris dengan
    # label berbeda tidak boleh dibuang diam-diam.
    training_frame = X_train.copy()
    training_frame["prognosis"] = y_train.to_numpy()
    training_frame = training_frame.drop_duplicates().reset_index(drop=True)
    X_train = training_frame[feature_cols].astype(int)
    y_train = training_frame["prognosis"].astype(str)

    print(f"Baris training awal: {raw_count}")
    print(f"Baris setelah deduplikasi persis: {len(training_frame)}")
    print(f"Duplikat dihapus: {raw_count - len(training_frame)}")

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train_enc)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "desease_model.joblib")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")

    print(f"Model tersimpan di: {MODELS_DIR}")
    print(f"Jumlah fitur: {len(feature_cols)} | Jumlah kelas: {len(encoder.classes_)}")
    print("Catatan: hasil model adalah prototipe edukasi, bukan alat diagnosis klinis.")
    return model, encoder, feature_cols


if __name__ == "__main__":
    train()
