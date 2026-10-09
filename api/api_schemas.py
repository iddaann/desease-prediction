"""Skema request/response untuk API prediksi edukatif."""
from pydantic import BaseModel, Field, field_validator


class SymptomRequest(BaseModel):
    symptoms: list[str] = Field(
        ...,
        min_length=1,
        max_length=132,
        description="Daftar gejala yang dijawab YA.",
        examples=[["itching", "skin_rash", "nodal_skin_eruptions"]],
    )
    asked: list[str] = Field(
        default_factory=list,
        max_length=132,
        description="Daftar gejala yang sudah pernah ditanyakan.",
    )

    @field_validator("symptoms", "asked")
    @classmethod
    def strip_and_deduplicate(cls, values: list[str]) -> list[str]:
        cleaned = []
        seen = set()
        for value in values:
            item = value.strip()
            if item and item not in seen:
                cleaned.append(item)
                seen.add(item)
        return cleaned


class DiseaseCandidate(BaseModel):
    penyakit: str
    # Output predict_proba; bukan probabilitas klinis terkalibrasi.
    probabilitas: float


class PredictionResponse(BaseModel):
    prediksi_utama: str
    kandidat: list[DiseaseCandidate]
    gejala_tidak_dikenali: list[str]
    next_question: str | None = None
    catatan: str | None = None
