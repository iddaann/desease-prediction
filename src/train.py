"""
Script training: latih Random Forest untuk prediksi penyakit dari gejala,
lalu simpan model + label encoder + daftar fitur ke folder models/.
"""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from preprocessing import load_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def train():
    X_train, y_train, X_test, y_test, feature_cols = load_dataset()

    # Encode label penyakit (string) -> angka, supaya konsisten dipakai
    # lintas model dan gampang di-serialize.
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train_enc)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "desease_model.joblib")
    joblib.dump(encoder, MODELS_DIR / "label_encoder.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")

    print(f"Model tersimpan di: {MODELS_DIR}")
    print(f"Jumlah fitur: {len(feature_cols)} | Jumlah kelas: {len(encoder.classes_)}")

    return model, encoder, feature_cols


if __name__ == "__main__":
    train()
