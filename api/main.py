import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from pewdict import predict_from_symptoms
from api.api_schemas import SymptomRequest, PredictionResponse

app = FastAPI(
    title="Disease Prediction API",
    description=(
        "API edukasi untuk eksplorasi gejala. Hasil bukan diagnosis "
        "dan tidak boleh digunakan untuk keputusan medis."
    ),
)

# Set CORS_ORIGINS pada environment deployment, dipisahkan koma.
# Default lokal hanya mengizinkan origin pengembangan yang umum.
_default_origins = "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8000,http://localhost:8000"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.get("/")
def read_root():
    return {
        "status": "ok",
        "message": "Disease Prediction API is running",
        "notice": "Educational use only; not a medical diagnosis.",
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SymptomRequest):
    if not request.symptoms:
        raise HTTPException(
            status_code=400,
            detail="Pilih atau masukkan setidaknya satu gejala terlebih dahulu.",
        )

    try:
        return predict_from_symptoms(
            symptoms=request.symptoms,
            asked=request.asked,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Artefak model atau dataset tidak tersedia. Periksa proses training/deployment.",
        ) from exc
