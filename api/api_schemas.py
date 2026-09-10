"""
Skema request/response untuk API prediksi penyakit.
"""

from pydantic import BaseModel, Field


class SymptomRequest(BaseModel):
    symptoms: list[str] = Field(
        ...,
        description="Daftar gejala yang sudah dijawab YA.",
        examples=[["itching", "skin_rash", "nodal_skin_eruptions"]],
    )

    asked: list[str] = Field(
        default_factory=list,
        description="Daftar gejala yang sudah pernah ditanyakan.",
    )


class DiseaseCandidate(BaseModel):
    penyakit: str
    probabilitas: float


class PredictionResponse(BaseModel):
    prediksi_utama: str
    kandidat: list[DiseaseCandidate]
    gejala_tidak_dikenali: list[str]
    next_question: str | None = None