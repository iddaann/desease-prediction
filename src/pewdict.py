from pathlib import Path

import joblib
import pandas as pd

from symptom_normalizer import normalize_symptoms


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MAX_FOLLOWUP_QUESTIONS = 5
STOP_PROBABILITY = 0.80
STOP_MARGIN = 0.45

_model = None
_encoder = None
_feature_cols = None
_training_data = None


def _load_artifacts():
    """Load semua artifact sekali dan simpan di memory."""
    global _model, _encoder, _feature_cols, _training_data

    if _model is None:
        _model = joblib.load(MODELS_DIR / "desease_model.joblib")
        _encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")
        _feature_cols = joblib.load(MODELS_DIR / "feature_columns.joblib")
        _training_data = pd.read_csv(DATA_DIR / "Training.csv")

        junk_cols = [
            col for col in _training_data.columns
            if col.lower().startswith("unnamed")
        ]
        if junk_cols:
            _training_data = _training_data.drop(columns=junk_cols)

    return _model, _encoder, _feature_cols, _training_data


def _entropy(probabilities):
    """Hitung entropy dengan aman."""
    values = [float(p) for p in probabilities if p > 0]
    if not values:
        return 0.0

    import math
    return -sum(p * math.log2(p) for p in values)


def _choose_next_question(candidates, symptoms, asked, feature_cols, training_data):
    """Memilih pertanyaan follow-up yang paling informatif."""
    if not candidates or len(asked) >= MAX_FOLLOWUP_QUESTIONS:
        return None

    top_probability = float(candidates[0]["probabilitas"])
    second_probability = (
        float(candidates[1]["probabilitas"])
        if len(candidates) > 1 else 0.0
    )
    margin = top_probability - second_probability

    if top_probability >= STOP_PROBABILITY or margin >= STOP_MARGIN:
        return None

    candidate_names = [c["penyakit"] for c in candidates]
    priors = {c["penyakit"]: float(c["probabilitas"]) for c in candidates}

    candidate_data = training_data[
        training_data["prognosis"].isin(candidate_names)
    ].copy()
    if candidate_data.empty:
        return None

    excluded = set(symptoms) | set(asked)
    prior_total = sum(priors.values())
    if prior_total <= 0:
        return None

    for disease in priors:
        priors[disease] /= prior_total

    current_entropy = _entropy(priors.values())
    best_symptom = None
    best_gain = 0.0

    for symptom in feature_cols:
        if symptom in excluded or symptom not in candidate_data.columns:
            continue

        prevalence_series = candidate_data.groupby("prognosis")[symptom].mean()
        prevalence = {
            disease: float(prevalence_series.get(disease, 0.0))
            for disease in candidate_names
        }
        values = list(prevalence.values())

        if not values or max(values) - min(values) < 0.10:
            continue

        p_yes = sum(
            priors[disease] * prevalence[disease]
            for disease in candidate_names
        )
        p_no = 1.0 - p_yes
        if p_yes <= 0.01 or p_no <= 0.01:
            continue

        posterior_yes_raw = {
            disease: priors[disease] * prevalence[disease]
            for disease in candidate_names
        }
        yes_total = sum(posterior_yes_raw.values())
        posterior_yes = [
            value / yes_total for value in posterior_yes_raw.values()
            if yes_total > 0
        ]

        posterior_no_raw = {
            disease: priors[disease] * (1.0 - prevalence[disease])
            for disease in candidate_names
        }
        no_total = sum(posterior_no_raw.values())
        posterior_no = [
            value / no_total for value in posterior_no_raw.values()
            if no_total > 0
        ]

        expected_entropy = (
            p_yes * _entropy(posterior_yes)
            + p_no * _entropy(posterior_no)
        )
        information_gain = current_entropy - expected_entropy
        discrimination = max(values) - min(values)
        score = information_gain + (0.05 * discrimination)

        if score > best_gain:
            best_gain = score
            best_symptom = symptom

    return best_symptom


def predict_from_symptoms(symptoms, asked=None, top_k=3):
    """Prediksi penyakit dan pilih pertanyaan follow-up secara adaptive."""
    if asked is None:
        asked = []

    model, encoder, feature_cols, training_data = _load_artifacts()

    # Input bahasa natural/alias dinormalisasi sebelum masuk ke model.
    valid_symptoms, unknown_symptoms = normalize_symptoms(symptoms, feature_cols)
    normalized_asked, _ = normalize_symptoms(asked, feature_cols)

    vector = pd.DataFrame(
        [[1 if column in valid_symptoms else 0 for column in feature_cols]],
        columns=feature_cols,
    )

    proba = model.predict_proba(vector)[0]
    top_idx = proba.argsort()[::-1][:top_k]

    candidates = [
        {
            "penyakit": encoder.classes_[index],
            "probabilitas": round(float(proba[index]), 4),
        }
        for index in top_idx
    ]

    next_question = _choose_next_question(
        candidates=candidates,
        symptoms=valid_symptoms,
        asked=normalized_asked,
        feature_cols=feature_cols,
        training_data=training_data,
    )

    return {
        "prediksi_utama": candidates[0]["penyakit"],
        "kandidat": candidates,
        "gejala_tidak_dikenali": unknown_symptoms,
        "next_question": next_question,
    }


if __name__ == "__main__":
    hasil = predict_from_symptoms(
        symptoms=["itching", "skin_rash", "nodal_skin_eruptions"]
    )
    print(hasil)
