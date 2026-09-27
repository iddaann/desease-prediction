import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
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

RED_FLAG_PATTERNS = {
    "sesak berat": ("sesak napas berat", "sesak napas berat atau sulit bernapas"),
    "nyeri dada berat": ("nyeri dada berat", "nyeri dada berat"),
    "sulit berbicara karena sesak": ("sulit berbicara karena sesak", "sulit berbicara karena sesak"),
    "pingsan": ("pingsan", "pingsan atau kehilangan kesadaran"),
    "kejang": ("kejang", "kejang"),
    "tidak sadar": ("tidak sadar", "penurunan kesadaran"),
    "perdarahan berat": ("perdarahan berat", "perdarahan berat"),
}


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


def _evaluate(y_true, pred, proba, encoder, prefix=""):
    labels = np.arange(len(encoder.classes_))
    report = classification_report(
        y_true,
        pred,
        labels=labels,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision_weighted": round(float(precision_score(y_true, pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, pred, average="weighted", zero_division=0)), 4),
        "precision_macro": round(float(precision_score(y_true, pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
        "classification_report": {
            label: {
                key: round(float(value), 4) if isinstance(value, (int, float)) else value
                for key, value in values.items()
            }
            for label, values in report.items()
            if isinstance(values, dict)
        },
        "confusion_matrix_labels": encoder.classes_.tolist(),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=labels).tolist(),
    }

    if proba is not None and len(encoder.classes_) >= 3:
        try:
            metrics["top3_accuracy"] = round(
                float(top_k_accuracy_score(y_true, proba, k=3, labels=labels)), 4
            )
        except ValueError:
            pass

    return metrics


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

    holdout = _evaluate(y_test_enc, pred, proba, encoder)

    cv_splits = min(5, min_class)
    cv_metrics = {}
    if cv_splits >= 2:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        cv_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        cv_accuracy = cross_val_score(cv_model, X, encoder.transform(y), cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(cv_model, X, encoder.transform(y), cv=cv, scoring="f1_macro")
        cv_metrics = {
            "folds": cv_splits,
            "accuracy_mean": round(float(cv_accuracy.mean()), 4),
            "accuracy_std": round(float(cv_accuracy.std()), 4),
            "f1_macro_mean": round(float(cv_f1.mean()), 4),
            "f1_macro_std": round(float(cv_f1.std()), 4),
        }

    external = None
    testing_path = BASE_DIR / "data" / "Testing.csv"
    if testing_path.exists():
        test_df = _clean(pd.read_csv(testing_path)).dropna(subset=["prognosis"])
        known_mask = test_df["prognosis"].astype(str).isin(set(encoder.classes_))
        unknown_classes = sorted(set(test_df.loc[~known_mask, "prognosis"].astype(str)))
        test_df = test_df.loc[known_mask].copy()
        if not test_df.empty:
            external_X = test_df.reindex(columns=features, fill_value=0).apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0)
            external_y = encoder.transform(test_df["prognosis"].astype(str))
            external_pred = model.predict(external_X)
            external_proba = model.predict_proba(external_X)
            external = {
                "dataset_name": testing_path.name,
                "samples_evaluated": len(test_df),
                "unknown_classes_skipped": unknown_classes,
                "metrics": _evaluate(external_y, external_pred, external_proba, encoder),
            }

    new_version = "disease-" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)
    joblib.dump(features, FEATURES_FILE)
    VERSION_FILE.write_text(new_version)

    metrics_payload = {
        "model_version": new_version,
        "dataset_name": path.name,
        "samples": len(df),
        "features": len(features),
        "classes": len(encoder.classes_),
        "evaluation_note": (
            "Model scores are classifier outputs on the supplied dataset; they are not calibrated "
            "clinical probabilities or medical diagnoses."
        ),
        "holdout": holdout,
        "cross_validation": cv_metrics,
        "external_test": external,
    }
    METRICS_FILE.write_text(json.dumps(metrics_payload, indent=2))

    return {
        "model_version": new_version,
        "dataset_name": path.name,
        "samples": len(df),
        "features": len(features),
        "classes": len(encoder.classes_),
        "metrics": metrics_payload,
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


def detect_red_flags(message):
    """Deteksi kata/frasa yang memerlukan peringatan keselamatan sebelum hasil ML."""
    text = str(message).lower()
    matches = []
    for _, (pattern, description) in RED_FLAG_PATTERNS.items():
        if pattern in text:
            matches.append(description)
    return matches


def predict_from_symptoms(symptoms, safety_alerts=None):
    model, encoder, features = load_bundle()
    normalized, unknown = normalize_symptoms(symptoms, features)
    safety_alerts = safety_alerts or []

    if not normalized:
        return {
            "needs_more_input": True,
            "status": "no_symptoms",
            "message": "Saya belum menemukan gejala yang cukup jelas. Coba ceritakan keluhanmu dengan kalimat biasa, misalnya: 'sejak kemarin demam, batuk kering, tenggorokan sakit, dan badan terasa lemas'.",
            "response_text": "Saya belum menangkap gejala yang cukup jelas dari pesanmu. Coba ceritakan apa yang kamu rasakan dengan bahasa sehari-hari. Kamu juga boleh menambahkan sejak kapan keluhannya muncul.",
            "recognized_symptoms": [],
            "unknown_symptoms": unknown,
            "candidates": [],
            "model_version": version(),
            "safety_alerts": safety_alerts,
        }

    frame = pd.DataFrame(0, index=[0], columns=features, dtype=float)
    for symptom in normalized:
        frame.at[0, symptom] = 1.0

    probabilities = model.predict_proba(frame)[0]
    order = np.argsort(probabilities)[::-1][: min(5, len(probabilities))]
    candidates = [
        {
            "disease": str(encoder.inverse_transform([i])[0]),
            "model_score": round(float(probabilities[i]), 4),
        }
        for i in order
    ]
    top = candidates[0]
    model_score = top["model_score"]
    low_model_score = model_score < 0.45 or len(normalized) < 2

    if safety_alerts:
        status = "safety_alert"
        message = "Ada tanda pada pesanmu yang sebaiknya mendapat perhatian medis segera."
        response_text = (
            "Sebelum melihat hasil model, saya ingin menekankan bahwa beberapa tanda yang kamu sebutkan "
            "dapat memerlukan penilaian medis segera. Jangan gunakan prediksi model ini untuk menentukan "
            "apakah kondisi tersebut aman ditunggu. Jika gejalanya berat, memburuk, atau sesuai dengan "
            "tanda darurat, cari pertolongan medis segera.\n\n"
            "Model tetap dapat menunjukkan kecocokan pola gejala untuk tujuan pembelajaran, tetapi hasilnya "
            "bukan diagnosis dan bukan pengganti pemeriksaan tenaga kesehatan."
        )
    elif low_model_score:
        status = "needs_clarification"
        message = (
            "Saya sudah mengenali beberapa gejala, tetapi informasinya belum cukup kuat "
            "untuk menyebut satu kemungkinan sebagai hasil utama."
        )
        response_text = (
            f"Saya sudah menangkap {len(normalized)} gejala dari ceritamu, tetapi model masih belum cukup yakin "
            f"untuk mengarah pada satu kondisi tertentu. Kecocokan model tertinggi saat ini adalah {top['disease']} "
            f"dengan skor model {model_score * 100:.1f}%.\n\n"
            "Kalau kamu mau, lanjutkan ceritanya. Misalnya jelaskan sejak kapan gejala muncul, seberapa berat, "
            "dan apakah ada keluhan lain. Saya akan menggabungkan informasi dari pesan-pesan sebelumnya."
        )
    else:
        status = "prediction"
        message = "Berdasarkan gejala yang dikenali, berikut kemungkinan teratas dari model."
        response_text = (
            f"Dari gejala yang kamu ceritakan, kecocokan pola tertinggi menurut model adalah {top['disease']} "
            f"dengan skor model {model_score * 100:.1f}%.\n\n"
            "Hasil ini adalah eksperimen klasifikasi berbasis dataset dan bukan diagnosis medis. "
            "Kamu bisa melanjutkan percakapan dengan menambahkan gejala lain agar input model lebih lengkap."
        )

    warnings = [
        "Skor model bukan probabilitas diagnosis dan tidak menunjukkan kemungkinan seseorang benar-benar memiliki penyakit tersebut.",
        "HealthPredict adalah project pembelajaran machine learning dan bukan alat diagnosis medis.",
    ]
    if low_model_score:
        warnings.insert(0, "Skor model relatif rendah atau gejala yang diberikan masih sedikit. Tambahkan konteks jika ingin menguji model lagi.")
    if safety_alerts:
        warnings.insert(0, "Terdapat tanda peringatan pada pesan. Prioritaskan pertolongan medis dan jangan mengandalkan hasil model.")

    return {
        "needs_more_input": False,
        "status": status,
        "message": message,
        "response_text": response_text,
        "recognized_symptoms": normalized,
        "unknown_symptoms": unknown,
        "disease": top["disease"],
        "model_score": model_score,
        "candidates": candidates,
        "recommendations": [
            "Untuk eksperimen model, kamu bisa menambahkan gejala lain pada pesan berikutnya.",
            "Perhatikan bahwa dataset dan model ini tidak mewakili seluruh kondisi klinis dunia nyata.",
            "Jika keluhan menetap, memburuk, atau mengganggu aktivitas, konsultasikan dengan tenaga kesehatan.",
        ],
        "warnings": warnings,
        "safety_alerts": safety_alerts,
        "model_version": version(),
    }
