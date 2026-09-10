import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Supaya bisa import modul dari folder src/
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from pewdict import predict_from_symptoms  # noqa: E402
from api.api_schemas import SymptomRequest, PredictionResponse  # noqa: E402

app = FastAPI(
    title="Disease Prediction API",
    description="API prediksi penyakit dari checklist gejala.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Disease Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: SymptomRequest):
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Daftar gejala tidak boleh kosong")

    hasil = predict_from_symptoms(request.symptoms)
    return hasil
