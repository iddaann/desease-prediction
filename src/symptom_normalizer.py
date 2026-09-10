"""Normalisasi nama gejala dari input pengguna ke feature model."""

from difflib import get_close_matches
import re
import unicodedata


# Alias umum bahasa Indonesia/Inggris.
# Alias dapat ditambah tanpa mengubah logic prediction.
SYMPTOM_ALIASES = {
    "demam": "fever",
    "panas": "fever",
    "badan panas": "fever",
    "suhu badan naik": "fever",
    "sakit kepala": "headache",
    "pusing": "headache",
    "nyeri kepala": "headache",
    "batuk": "cough",
    "mual": "nausea",
    "muntah": "vomiting",
    "gatal": "itching",
    "ruam kulit": "skin_rash",
    "badan pegal": "muscle_pain",
    "pegal": "muscle_pain",
    "pegal pegal": "muscle_pain",
    "pegal-pegal": "muscle_pain",
    "nyeri otot": "muscle_pain",
    "sakit otot": "muscle_pain",
    "otot sakit": "muscle_pain",
    "sakit pinggang": "back_pain",
    "nyeri pinggang": "back_pain",
    "pinggang sakit": "back_pain",
    "pinggang nyeri": "back_pain",
    "sakit punggung": "back_pain",
    "nyeri punggung": "back_pain",
    "punggung sakit": "back_pain",
    "back pain": "back_pain",
    "sesak napas": "breathlessness",
    "sesak nafas": "breathlessness",
    "sulit bernapas": "breathlessness",
    "sulit bernafas": "breathlessness",
    "napas sesak": "breathlessness",
    "berak darah": "bloody_stool",
    "tinja berdarah": "bloody_stool",
    "bab berdarah": "bloody_stool",
}


def _clean_text(value: str) -> str:
    """Bersihkan teks agar pencocokan lebih konsisten."""
    value = unicodedata.normalize("NFKC", str(value)).lower().strip()
    value = value.replace("_", " ")
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _feature_to_text(feature: str) -> str:
    """Ubah nama feature seperti muscle_pain menjadi teks biasa."""
    return _clean_text(feature)


def normalize_symptom(symptom: str, feature_cols: list[str]) -> str | None:
    """Kembalikan nama feature model yang paling sesuai, atau None."""
    cleaned = _clean_text(symptom)

    if not cleaned:
        return None

    # 1. Input sudah berupa nama feature model.
    feature_lookup = {
        _feature_to_text(feature): feature
        for feature in feature_cols
    }

    if cleaned in feature_lookup:
        return feature_lookup[cleaned]

    # 2. Alias bahasa natural.
    alias = SYMPTOM_ALIASES.get(cleaned)
    if alias and alias in feature_cols:
        return alias

    # 3. Fuzzy matching hanya jika cukup dekat.
    # Ini membantu typo sederhana seperti "headake" -> "headache".
    close = get_close_matches(
        cleaned,
        list(feature_lookup.keys()),
        n=1,
        cutoff=0.88,
    )

    if close:
        return feature_lookup[close[0]]

    return None


def normalize_symptoms(
    symptoms: list[str],
    feature_cols: list[str],
) -> tuple[list[str], list[str]]:
    """Normalisasi banyak gejala dan pisahkan gejala yang tidak dikenali."""
    normalized = []
    unknown = []

    for symptom in symptoms:
        result = normalize_symptom(symptom, feature_cols)

        if result is None:
            unknown.append(symptom)
        elif result not in normalized:
            normalized.append(result)

    return normalized, unknown
