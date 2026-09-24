import json
import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.auth import create_session, get_current_user, require_admin
from api.chat_schemas import ChatRequest, RetrainRequest
from api.db import connect, decrypt_text, encrypt_text, hash_password, init_db, utcnow, verify_password
from api.disease_ml import available, feature_list, metrics as model_metrics, predict_from_symptoms, extract_symptoms, train, version

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
init_db()

app = FastAPI(
    title="HealthPredict API",
    version="2.0.0",
    description="Chatbot prediksi penyakit berbasis gejala dengan machine learning.",
)

origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": available(),
        "model_version": version(),
    }

@app.get("/")
def root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/api/symptoms")
def symptoms():
    return {"symptoms": sorted(feature_list())}

@app.post("/api/chat")
def chat(payload: ChatRequest):
    session_id = payload.session_id or secrets.token_urlsafe(18)

    # Ambil gejala dari percakapan sebelumnya agar follow-up seperti
    # "batuknya juga ada" atau "terus bagaimana?" tetap punya konteks.
    previous_symptoms = []
    previous_result = None
    with connect() as db:
        rows = db.execute(
            "SELECT symptoms,result FROM consultations WHERE session_id=? ORDER BY id DESC LIMIT 12",
            (session_id,),
        ).fetchall()
    for row in rows:
        try:
            previous_symptoms.extend(json.loads(decrypt_text(row["symptoms"])))
            if previous_result is None:
                previous_result = json.loads(decrypt_text(row["result"]))
        except (json.JSONDecodeError, TypeError):
            continue

    extracted, unknown = extract_symptoms(payload.message, payload.symptoms)
    merged = list(dict.fromkeys(previous_symptoms + extracted))

    # Jika user hanya melanjutkan percakapan tanpa menambahkan gejala,
    # berikan jawaban berbasis hasil terakhir alih-alih memulai prediksi dari nol.
    if not extracted and previous_result and previous_result.get("response_text"):
        result = dict(previous_result)
        result["response_text"] = (
            "Tentu, kita bisa lanjut dari pembahasan sebelumnya.\n\n"
            + previous_result["response_text"]
        )
        result["recognized_symptoms"] = merged
        result["session_id"] = session_id
    else:
        result = predict_from_symptoms(merged)
        result["recognized_symptoms"] = merged
        result["unknown_symptoms"] = unknown
        result["session_id"] = session_id

    with connect() as db:
        db.execute(
            """
            INSERT INTO consultations(session_id,message,symptoms,result,model_version,created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (
                session_id,
                encrypt_text(payload.message),
                encrypt_text(json.dumps(merged, ensure_ascii=False)),
                encrypt_text(json.dumps(result, ensure_ascii=False)),
                result.get("model_version", version()),
                utcnow(),
            ),
        )

    return result

@app.get("/api/history/{session_id}")
def history(session_id: str):
    if len(session_id) > 100:
        raise HTTPException(status_code=400, detail="Session ID tidak valid.")
    with connect() as db:
        rows = db.execute(
            "SELECT id,message,symptoms,result,model_version,created_at FROM consultations WHERE session_id=? ORDER BY id ASC LIMIT 100",
            (session_id,),
        ).fetchall()

    output = []
    for row in rows:
        output.append({
            "id": row["id"],
            "message": decrypt_text(row["message"]),
            "symptoms": json.loads(decrypt_text(row["symptoms"])),
            "result": json.loads(decrypt_text(row["result"])),
            "model_version": row["model_version"],
            "created_at": row["created_at"],
        })
    return output

@app.post("/auth/login")
def login(payload: dict):
    email = str(payload.get("email", "")).strip()
    password = str(payload.get("password", ""))
    if not email or not password:
        raise HTTPException(status_code=422, detail="Email dan password wajib diisi.")
    with connect() as db:
        user = db.execute(
            "SELECT id,name,email,password_hash,role FROM users WHERE lower(email)=lower(?)",
            (email,),
        ).fetchone()
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email atau password salah.")
    return {
        "access_token": create_session(user["id"]),
        "user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]},
    }

@app.get("/auth/me")
def me(user=Depends(get_current_user)):
    return user

@app.get("/admin/dashboard")
def admin_dashboard(user=Depends(require_admin)):
    with connect() as db:
        consultations = db.execute("SELECT COUNT(*) FROM consultations").fetchone()[0]
        sessions = db.execute("SELECT COUNT(DISTINCT session_id) FROM consultations").fetchone()[0]
        runs = db.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
    return {
        "consultations": consultations,
        "sessions": sessions,
        "retrain_runs": runs,
        "model_version": version(),
        "model_ready": available(),
        "metrics": model_metrics(),
    }

@app.get("/admin/model-metrics")
def admin_metrics(user=Depends(require_admin)):
    return model_metrics() or {"model_version": version(), "message": "Model belum dilatih."}

@app.post("/admin/retrain")
def retrain(payload: RetrainRequest, user=Depends(require_admin)):
    try:
        result = train(payload.dataset_path)
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
                (version(), "failed", json.dumps({"error": str(exc)}), payload.dataset_path, utcnow()),
            )
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/admin/consultations")
def admin_consultations(user=Depends(require_admin)):
    with connect() as db:
        rows = db.execute(
            "SELECT id,session_id,message,result,created_at FROM consultations ORDER BY id DESC LIMIT 100"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "session_id": row["session_id"],
            "message": decrypt_text(row["message"]),
            "result": json.loads(decrypt_text(row["result"])),
            "created_at": row["created_at"],
        }
        for row in rows
    ]
