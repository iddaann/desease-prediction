from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

_model = None
_encoder = None
_feature_cols = None
_training_data = None


def _load_artifacts():
    """
    Load model, encoder, feature columns, dan dataset training.
    Semua artifact hanya dimuat sekali lalu disimpan di memory.
    """

    global _model, _encoder, _feature_cols, _training_data

    if _model is None:

        _model = joblib.load(
            MODELS_DIR / "desease_model.joblib"
        )

        _encoder = joblib.load(
            MODELS_DIR / "label_encoder.joblib"
        )

        _feature_cols = joblib.load(
            MODELS_DIR / "feature_columns.joblib"
        )

        training_path = DATA_DIR / "Training.csv"

        _training_data = pd.read_csv(training_path)

        # Hapus kolom sampah jika ada
        junk_cols = [
            col
            for col in _training_data.columns
            if col.lower().startswith("unnamed")
        ]

        if junk_cols:
            _training_data = _training_data.drop(
                columns=junk_cols
            )

    return (
        _model,
        _encoder,
        _feature_cols,
        _training_data,
    )


def _choose_next_question(
    candidates: list[dict],
    symptoms: list[str],
    asked: list[str],
    feature_cols: list[str],
    training_data: pd.DataFrame,
) -> str | None:
    """
    Memilih gejala berikutnya berdasarkan perbedaan karakteristik
    antar kandidat penyakit.

    Gejala yang sudah dijawab atau sudah ditanyakan tidak akan
    dipilih lagi.
    """

    # Kandidat penyakit
    candidate_names = [
        candidate["penyakit"]
        for candidate in candidates
    ]

    # Gejala yang tidak boleh ditanyakan lagi
    excluded = set(symptoms) | set(asked)

    # Ambil data training hanya untuk kandidat penyakit
    candidate_data = training_data[
        training_data["prognosis"].isin(candidate_names)
    ]

    if candidate_data.empty:
        return None

    best_symptom = None
    best_score = -1

    for symptom in feature_cols:

        # Jangan tanyakan gejala yang sudah pernah digunakan
        if symptom in excluded:
            continue

        if symptom not in candidate_data.columns:
            continue

        # Persentase kemunculan gejala pada masing-masing penyakit
        prevalence = (
            candidate_data
            .groupby("prognosis")[symptom]
            .mean()
        )

        if len(prevalence) < 2:
            continue

        # Cari perbedaan terbesar antar kandidat
        score = float(
            prevalence.max() - prevalence.min()
        )

        # Sedikit mempertimbangkan probabilitas kandidat
        if score > best_score:
            best_score = score
            best_symptom = symptom

    return best_symptom


def predict_from_symptoms(
    symptoms: list[str],
    asked: list[str] | None = None,
    top_k: int = 3,
) -> dict:
    """
    Prediksi penyakit berdasarkan gejala.

    Parameters
    ----------
    symptoms:
        Gejala yang dijawab YA oleh user.

    asked:
        Gejala yang sudah pernah ditanyakan kepada user.

    top_k:
        Jumlah kandidat penyakit yang dikembalikan.
    """

    if asked is None:
        asked = []

    (
        model,
        encoder,
        feature_cols,
        training_data,
    ) = _load_artifacts()

    # -----------------------------------
    # 1. Validasi gejala
    # -----------------------------------

    valid_symptoms = [
        symptom
        for symptom in symptoms
        if symptom in feature_cols
    ]

    unknown_symptoms = [
        symptom
        for symptom in symptoms
        if symptom not in feature_cols
    ]

    # -----------------------------------
    # 2. Buat vector input model
    # -----------------------------------

    vector = pd.DataFrame(
        [
            [
                1 if column in valid_symptoms else 0
                for column in feature_cols
            ]
        ],
        columns=feature_cols,
    )

    # -----------------------------------
    # 3. Prediksi probabilitas
    # -----------------------------------

    proba = model.predict_proba(vector)[0]

    top_idx = proba.argsort()[::-1][:top_k]

    candidates = [
        {
            "penyakit": encoder.classes_[index],
            "probabilitas": round(
                float(proba[index]),
                4,
            ),
        }
        for index in top_idx
    ]

    # -----------------------------------
    # 4. Tentukan pertanyaan berikutnya
    # -----------------------------------

    next_question = _choose_next_question(
        candidates=candidates,
        symptoms=valid_symptoms,
        asked=asked,
        feature_cols=feature_cols,
        training_data=training_data,
    )

    # -----------------------------------
    # 5. Kembalikan hasil
    # -----------------------------------

    return {
        "prediksi_utama": candidates[0]["penyakit"],
        "kandidat": candidates,
        "gejala_tidak_dikenali": unknown_symptoms,
        "next_question": next_question,
    }


if __name__ == "__main__":

    hasil = predict_from_symptoms(
        symptoms=[
            "itching",
            "skin_rash",
            "nodal_skin_eruptions",
        ]
    )

    print(hasil)