import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "hospital"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VERSION_FILE = MODEL_DIR / "version.txt"
METRICS_FILE = MODEL_DIR / "metrics.json"
FEATURES = ["Age", "Gender", "Blood Type", "Admission Type", "Medication", "Test Results"]
CAT = FEATURES[1:]

def version():
    return VERSION_FILE.read_text().strip() if VERSION_FILE.exists() else "hospital-model-not-trained"

def available():
    return all((MODEL_DIR / n).exists() for n in ("classifier.joblib", "stay.joblib", "cost.joblib"))

def pipeline(model):
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT), ("num", "passthrough", ["Age"])])
    return Pipeline([("prep", prep), ("model", model)])

def predict(patient):
    if not available():
        return {"medical_condition": None, "confidence_score": None, "candidates": [], "estimated_length_of_stay_days": None, "estimated_cost": None, "warnings": ["Model SRS rumah sakit belum dilatih. Sediakan dataset sesuai schema SRS lalu jalankan retrain."], "model_version": version()}
    frame = pd.DataFrame([{"Age": patient["age"], "Gender": patient["gender"], "Blood Type": patient["blood_type"], "Admission Type": patient["admission_type"], "Medication": patient["medication"], "Test Results": patient["test_results"]}])
    clf = joblib.load(MODEL_DIR / "classifier.joblib")
    stay = joblib.load(MODEL_DIR / "stay.joblib")
    cost = joblib.load(MODEL_DIR / "cost.joblib")
    probabilities = clf.predict_proba(frame)[0]
    order = np.argsort(probabilities)[::-1][:3]
    candidates = [{"medical_condition": str(clf.classes_[i]), "probability": round(float(probabilities[i]), 4)} for i in order]
    return {"medical_condition": candidates[0]["medical_condition"], "confidence_score": candidates[0]["probability"], "candidates": candidates, "estimated_length_of_stay_days": round(max(0.0, float(stay.predict(frame)[0])), 2), "estimated_cost": round(max(0.0, float(cost.predict(frame)[0])), 2), "warnings": [], "model_version": version()}

def train(dataset_path):
    path = (BASE_DIR / dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    required = set(FEATURES + ["Medical Condition", "Length of Stay", "Billing Amount"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Kolom dataset belum sesuai SRS: " + ", ".join(sorted(missing)))
    df = df.dropna(subset=list(required))
    X = df[FEATURES]
    y = df["Medical Condition"]
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=.2, random_state=42, stratify=y)
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    clf = pipeline(RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    cls_metrics = {"accuracy": round(float(accuracy_score(y_test, pred)), 4), "precision": round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4), "recall": round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 4), "f1_score": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4), "cv_accuracy": round(float(cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()), 4)}
    stay = pipeline(RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1))
    cost = pipeline(RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1))
    stay.fit(x_train, df["Length of Stay"].iloc[train_idx])
    cost.fit(x_train, df["Billing Amount"].iloc[train_idx])
    stay_pred, cost_pred = stay.predict(x_test), cost.predict(x_test)
    reg_metrics = {"stay_rmse": round(float(np.sqrt(mean_squared_error(df["Length of Stay"].iloc[test_idx], stay_pred))), 4), "stay_mae": round(float(mean_absolute_error(df["Length of Stay"].iloc[test_idx], stay_pred)), 4), "stay_r2": round(float(r2_score(df["Length of Stay"].iloc[test_idx], stay_pred)), 4), "cost_rmse": round(float(np.sqrt(mean_squared_error(df["Billing Amount"].iloc[test_idx], cost_pred))), 4), "cost_mae": round(float(mean_absolute_error(df["Billing Amount"].iloc[test_idx], cost_pred)), 4), "cost_r2": round(float(r2_score(df["Billing Amount"].iloc[test_idx], cost_pred)), 4)}
    joblib.dump(clf, MODEL_DIR / "classifier.joblib")
    joblib.dump(stay, MODEL_DIR / "stay.joblib")
    joblib.dump(cost, MODEL_DIR / "cost.joblib")
    new_version = "hospital-" + pd.Timestamp.utcnow().strftime("%Y%m%d%H%M%S")
    VERSION_FILE.write_text(new_version)
    result = {"classification": cls_metrics, "regression": reg_metrics, "model_version": new_version, "dataset_name": path.name}
    METRICS_FILE.write_text(json.dumps(result, indent=2))
    return result

def metrics():
    return json.loads(METRICS_FILE.read_text()) if METRICS_FILE.exists() else None
