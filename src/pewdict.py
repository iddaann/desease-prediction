"""
Modul prediksi: menerima daftar gejala (nama-nama string, sesuai checklist
di frontend) lalu mengembalikan penyakit yang diprediksi model.
"""
from pathlib import Path
import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_model = None
_encoder = None
_feature_cols = None


def _load_artifacts():
    """Load model/encoder/daftar fitur sekali saja (di-cache di memori)."""
    global _model, _encoder, _feature_cols
    if _model is None:
        _model = joblib.load(MODELS_DIR / "desease_model.joblib")
        _encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
        _feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
    return _model, _encoder, _feature_cols


def predict_from_symptoms(symptoms: list[str], top_k: int = 3) -> dict:
    """
    Args:
        symptoms: daftar nama gejala yang dicentang user, mis.
                  ["itching", "skin_rash", "high_fever"]
        top_k: jumlah kandidat penyakit teratas yang mau ditampilkan
               (berguna karena beberapa kombinasi gejala bisa ambigu)

    Returns:
        dict berisi prediksi utama + top-k kandidat beserta probabilitasnya.
    """
    model, encoder, feature_cols = _load_artifacts()

    # Validasi: gejala yang tidak dikenali diabaikan (dilaporkan balik)
    valid_symptoms = [s for s in symptoms if s in feature_cols]
    unknown_symptoms = [s for s in symptoms if s not in feature_cols]

    # Bangun vektor 0/1 sesuai urutan kolom saat training
    vector = pd.DataFrame(
        [[1 if col in valid_symptoms else 0 for col in feature_cols]],
        columns=feature_cols,
    )

    proba = model.predict_proba(vector)[0]
    top_idx = proba.argsort()[::-1][:top_k]

    candidates = [
        {"penyakit": encoder.classes_[i], "probabilitas": round(float(proba[i]), 4)}
        for i in top_idx
    ]

    return {
        "prediksi_utama": candidates[0]["penyakit"],
        "kandidat": candidates,
        "gejala_tidak_dikenali": unknown_symptoms,
    }


if __name__ == "__main__":
    # Contoh pemakaian manual
    hasil = predict_from_symptoms(["itching", "skin_rash", "nodal_skin_eruptions"])
    print(hasil)
