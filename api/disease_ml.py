import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from src.symptom_normalizer import extract_symptoms_from_text, normalize_symptoms

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "disease"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "model.joblib"
ENCODER_FILE = MODEL_DIR / "label_encoder.joblib"
FEATURES_FILE = MODEL_DIR / "feature_columns.joblib"
METRICS_FILE = MODEL_DIR / "metrics.json"
VERSION_FILE = MODEL_DIR / "version.txt"

def version():
    return VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "disease-model-not-trained"

def available():
    return all(p.exists() for p in (MODEL_FILE, ENCODER_FILE, FEATURES_FILE))

def load_bundle():
    if not available():
        raise RuntimeError("Model penyakit belum dilatih. Jalankan retrain terlebih dahulu.")
    return (
        joblib.load(MODEL_FILE),
        joblib.load(ENCODER_FILE),
        joblib.load(FEATURES_FILE),
    )

def _clean(df):
    return df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")

def _load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = _clean(pd.read_csv(path))
    if "prognosis" not in df.columns:
        raise ValueError("Dataset harus memiliki kolom target 'prognosis'.")
    df = df.dropna(subset=["prognosis"]).drop_duplicates().reset_index(drop=True)
    features = [c for c in df.columns if c != "prognosis"]
    if len(df) < 20:
        raise ValueError("Dataset terlalu kecil. Gunakan dataset gejala-penyakit yang lengkap.")
    if df["prognosis"].nunique() < 2:
        raise ValueError("Dataset harus memiliki minimal dua kelas penyakit.")
    return df, features

def train(dataset_path="data/Training.csv"):
    path = (BASE_DIR / dataset_path).resolve()
    df, features = _load_dataset(path)
    X = df[features].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df["prognosis"].astype(str)

    min_class = int(y.value_counts().min())
    if min_class < 2:
        raise ValueError("Setiap penyakit harus memiliki minimal dua contoh untuk evaluasi.")
    test_size = max(0.2, 2 / len(df))
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    model.fit(x_train, y_train_enc)
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test_enc, pred)), 4),
        "precision": round(float(precision_score(y_test_enc, pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test_enc, pred, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test_enc, pred, average="weighted", zero_division=0)), 4),
    }

    cv_splits = min(5, min_class)
    if cv_splits >= 2:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        cv_model = RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )
        cv_scores = cross_val_score(cv_model, X, encoder.transform(y), cv=cv, scoring="accuracy")
        metrics["cv_accuracy"] = round(float(cv_scores.mean()), 4)
        metrics["cv_accuracy_std"] = round(float(cv_scores.std()), 4)

    if len(encoder.classes_) >= 3:
        metrics["top3_accuracy"] = round(float(top_k_accuracy_score(
            y_test_enc, proba, k=3, labels=np.arange(len(encoder.classes_))
        )), 4)

    new_version = "disease-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    joblib.dump(features, FEATURES_FILE)
    VERSION_FILE.write_text(new_version)
    METRICS_FILE.write_text(json.dumps({
        "model_version": new_version,
        "dataset_name": path.name,
        "samples": len(df),
        "features": len(features),
        "classes": len(encoder.classes_),
        "metrics": metrics,
    }, indent=2))

    return {
        "model_version": new_version,
        "dataset_name": path.name,
        "samples": len(df),
        "features": len(features),
        "classes": len(encoder.classes_),
        "metrics": metrics,
    }

def metrics():
    return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else None

def feature_list():
    if not available():
        return []
    return joblib.load(FEATURES_FILE)

def extract_symptoms(message, selected=None):
    """Ekstrak gejala dari kalimat natural dan pilihan eksplisit pengguna."""
    features = feature_list()
    selected = selected or []
    normalized_selected, unknown = normalize_symptoms(selected, features)
    extracted, _ = extract_symptoms_from_text(str(message), features)

    candidates = list(normalized_selected)
    for feature in extracted:
        if feature not in candidates:
            candidates.append(feature)

    return sorted(candidates), unknown


def predict_from_symptoms(symptoms):
    model, encoder, features = load_bundle()
    normalized, unknown = normalize_symptoms(symptoms, features)

    if not normalized:
        return {
            "needs_more_input": True,
            "status": "no_symptoms",
            "message": "Saya belum menemukan gejala yang cukup jelas. Coba ceritakan keluhan secara spesifik, misalnya 'demam sejak kemarin, batuk kering, tenggorokan sakit, dan badan lemas'.",
            "recognized_symptoms": [],
            "unknown_symptoms": unknown,
            "candidates": [],
            "model_version": version(),
        }

    frame = pd.DataFrame(0, index=[0], columns=features, dtype=float)
    for symptom in normalized:
        frame.at[0, symptom] = 1.0

    probabilities = model.predict_proba(frame)[0]
    order = np.argsort(probabilities)[::-1][: min(5, len(probabilities))]
    candidates = [
        {
            "disease": str(encoder.inverse_transform([i])[0]),
            "probability": round(float(probabilities[i]), 4),
        }
        for i in order
    ]
    top = candidates[0]
    confidence = top["probability"]

    # Random Forest probability pada dataset gejala yang sparse tidak selalu
    # berarti kepastian klinis. Gunakan ambang konservatif agar chatbot tidak
    # terdengar seperti memberikan diagnosis ketika bukti gejalanya lemah.
    low_confidence = confidence < 0.45 or len(normalized) < 2

    if low_confidence:
        status = "needs_clarification"
        message = (
            "Saya sudah mengenali beberapa gejala, tetapi informasinya belum cukup "
            "kuat untuk menyebut satu kemungkinan sebagai hasil utama. "
            "Coba tambahkan gejala lain, misalnya sejak kapan keluhan muncul, "
            "apakah ada demam, batuk, nyeri, mual, muntah, atau perubahan lain yang kamu rasakan."
        )
    else:
        status = "prediction"
        message = "Berdasarkan gejala yang dikenali, berikut kemungkinan teratas dari model."

    warnings = ["Hasil ini adalah prediksi berbasis gejala, bukan diagnosis medis."]
    if low_confidence:
        warnings.insert(0, "Confidence model relatif rendah. Tambahkan gejala agar hasil lebih informatif.")
    if any(s in normalized for s in ("breathlessness", "chest_pain")):
        warnings.append("Jika sesak berat, nyeri dada berat, sulit berbicara karena sesak, atau kondisi memburuk, cari pertolongan medis segera.")

    return {
        "needs_more_input": False,
        "status": status,
        "message": message,
        "recognized_symptoms": normalized,
        "unknown_symptoms": unknown,
        "disease": top["disease"],
        "confidence_score": confidence,
        "candidates": candidates,
        "recommendations": [
            "Perhatikan perkembangan dan perubahan gejala.",
            "Tambahkan gejala lain yang kamu alami agar pencocokan model lebih lengkap.",
            "Jika keluhan menetap, memburuk, atau mengganggu aktivitas, konsultasikan dengan tenaga kesehatan.",
        ],
        "warnings": warnings,
        "model_version": version(),
    }

