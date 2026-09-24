import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from api.db import init_db, connect, hash_password, verify_password, utcnow
from api.auth import create_session, get_current_user, require_admin
from api.srs_schemas import LoginRequest, PatientInput, RetrainRequest
from api.srs_ml import predict as predict_hospital, train as train_hospital, metrics as latest_metrics, version as model_version

init_db()

app = FastAPI(
    title="Disease Prediction & Health Recommendation API",
    description="REST API sesuai SRS: input pasien, klasifikasi, estimasi lama rawat/biaya, rekomendasi, riwayat, evaluasi dan retrain."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def recommendations(patient, result):
    items = []
    if result["confidence_score"] is not None and result["confidence_score"] < 0.60:
        items.append("Confidence prediksi masih rendah; hasil perlu dikonfirmasi dengan pemeriksaan tenaga medis.")
    if patient.admission_type.lower() in {"emergency", "darurat", "urgent", "mendesak"}:
        items.append("Prioritaskan asesmen tenaga medis sesuai prosedur kegawatdaruratan.")
    if result["estimated_length_of_stay_days"] is not None:
        items.append("Gunakan estimasi lama rawat sebagai bahan perencanaan tempat tidur dan sumber daya, bukan keputusan klinis tunggal.")
    items.append("Hasil sistem adalah alat bantu skrining dan perencanaan, bukan pengganti diagnosis resmi.")
    return items

@app.get("/")
def root():
    return {"status": "ok", "service": "Disease Prediction & Health Recommendation API", "model_version": model_version()}

@app.post("/auth/login")
def login(payload: LoginRequest):
    with connect() as db:
        user = db.execute("SELECT id,name,email,password_hash,role FROM users WHERE lower(email)=lower(?)", (payload.email,)).fetchone()
    if not user or not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email atau password salah")
    token = create_session(user["id"])
    return {"access_token": token, "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]}}

@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    return user

@app.post("/predict")
def predict(patient: PatientInput, user=Depends(get_current_user)):
    result = predict_hospital(patient.model_dump())
    result["recommendations"] = recommendations(patient, result)
    payload = patient.model_dump()
    with connect() as db:
        cur = db.execute(
            "INSERT INTO predictions(user_id,patient_payload,result_payload,model_version,created_at) VALUES(?,?,?,?,?)",
            (user["id"], json.dumps(payload), json.dumps(result), result["model_version"], utcnow()),
        )
        record_id = cur.lastrowid
    result["id"] = record_id
    return result

@app.get("/predictions")
def history(user=Depends(get_current_user), limit: int = 50):
    limit = max(1, min(limit, 200))
    with connect() as db:
        rows = db.execute(
            "SELECT id,patient_payload,result_payload,model_version,created_at FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user["id"], limit),
        ).fetchall()
    return [{"id": r["id"], "patient": json.loads(r["patient_payload"]), "result": json.loads(r["result_payload"]), "model_version": r["model_version"], "created_at": r["created_at"]} for r in rows]

@app.get("/admin/model-metrics")
def model_metrics(user=Depends(require_admin)):
    return latest_metrics() or {"model_version": model_version(), "message": "Belum ada model SRS rumah sakit yang dilatih."}

@app.get("/admin/dashboard")
def admin_dashboard(user=Depends(require_admin)):
    with connect() as db:
        users = db.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        predictions = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        runs = db.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
    return {"users": users, "predictions": predictions, "retrain_runs": runs, "model_version": model_version(), "metrics": latest_metrics()}

@app.post("/admin/retrain")
def retrain(payload: RetrainRequest, user=Depends(require_admin)):
    try:
        result = train_hospital(payload.dataset_path)
        with connect() as db:
            db.execute(
                "INSERT INTO model_runs(model_version,status,metrics_payload,dataset_name,created_at) VALUES(?,?,?,?,?)",
                (result["model_version"], "success", json.dumps(result), result["dataset_name"], utcnow()),
            )
        return {"status": "success", **result}
    except Exception as exc:
        with connect() as db:
            db.execute(
                "INSERT INTO model_runs(model_version,status,metrics_payload,dataset_name,created_at) VALUES(?,?,?,?,?)",
                (model_version(), "failed", json.dumps({"error": str(exc)}), payload.dataset_path, utcnow()),
            )
        raise HTTPException(status_code=400, detail=str(exc))
