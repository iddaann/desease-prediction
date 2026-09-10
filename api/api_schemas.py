"""
Skema request/response untuk API prediksi penyakit.
"""
from pydantic import BaseModel, Field


class SymptomRequest(BaseModel):
    symptoms: list[str] = Field(
        ...,
        description="Daftar nama gejala yang dicentang user, mis. ['itching', 'high_fever']",
        examples=[["itching", "skin_rash", "high_fever"]],
    )


class DiseaseCandidate(BaseModel):
    penyakit: str
    probabilitas: float


class PredictionResponse(BaseModel):
    prediksi_utama: str
    kandidat: list[DiseaseCandidate]
    gejala_tidak_dikenali: list[str]
