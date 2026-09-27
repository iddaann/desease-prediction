"""Normalisasi nama gejala dari input pengguna ke feature model."""

from difflib import get_close_matches
import re
import unicodedata


# Alias umum bahasa Indonesia/Inggris.
# Alias dapat ditambah tanpa mengubah logic prediction.
SYMPTOM_ALIASES = {
    "demam": "fever", "panas": "fever", "badan panas": "fever", "suhu badan naik": "fever",
    "sakit kepala": "headache", "pusing": "headache", "nyeri kepala": "headache",
    "batuk": "cough", "batuk kering": "dry_cough",
    "mual": "nausea", "muntah": "vomiting",
    "gatal": "itching", "ruam": "skin_rash", "ruam kulit": "skin_rash",
    "pegal": "muscle_pain", "badan pegal": "muscle_pain", "pegal pegal": "muscle_pain",
    "pegal-pegal": "muscle_pain", "nyeri otot": "muscle_pain", "sakit otot": "muscle_pain",
    "sakit sendi": "joint_pain", "nyeri sendi": "joint_pain", "sendi sakit": "joint_pain",
    "sakit perut": "stomach_pain", "nyeri perut": "stomach_pain", "perut sakit": "stomach_pain",
    "perut kembung": "stomach_pain", "maag": "acidity",
    "diare": "diarrhoea", "mencret": "diarrhoea", "sembelit": "constipation",
    "sakit tenggorokan": "throat_irritation", "tenggorokan sakit": "throat_irritation",
    "tenggorokan gatal": "throat_irritation", "sulit menelan": "difficulty_swallowing",
    "pilek": "runny_nose", "hidung berair": "runny_nose", "hidung tersumbat": "congestion",
    "bersin": "continuous_sneezing", "sering bersin": "continuous_sneezing",
    "sesak napas": "breathlessness", "sesak nafas": "breathlessness",
    "sulit bernapas": "breathlessness", "sulit bernafas": "breathlessness",
    "napas sesak": "breathlessness", "napas pendek": "breathlessness",
    "nyeri dada": "chest_pain", "dada sakit": "chest_pain",
    "jantung berdebar": "palpitations", "berdebar": "palpitations",
    "pusing berputar": "spinning_movements", "vertigo": "spinning_movements",
    "lemas": "fatigue", "kelelahan": "fatigue", "mudah lelah": "fatigue",
    "menggigil": "chills", "badan menggigil": "chills",
    "keringat berlebih": "sweating", "banyak berkeringat": "sweating",
    "nafsu makan turun": "loss_of_appetite", "tidak nafsu makan": "loss_of_appetite",
    "berat badan turun": "weight_loss", "berat badan naik": "weight_gain",
    "sering haus": "excessive_thirst", "haus berlebihan": "excessive_thirst",
    "sering kencing": "polyuria", "sering buang air kecil": "polyuria",
    "kencing sakit": "painful_urination", "buang air kecil sakit": "painful_urination",
    "urin berdarah": "blood_in_urine", "kencing berdarah": "blood_in_urine",
    "mata merah": "redness_of_eyes", "mata gatal": "itching",
    "penglihatan kabur": "blurred_and_distorted_vision", "mata kabur": "blurred_and_distorted_vision",
    "sakit pinggang": "back_pain", "nyeri pinggang": "back_pain",
    "pinggang sakit": "back_pain", "pinggang nyeri": "back_pain",
    "sakit punggung": "back_pain", "nyeri punggung": "back_pain", "back pain": "back_pain",
    "berak darah": "bloody_stool", "tinja berdarah": "bloody_stool", "bab berdarah": "bloody_stool",
    "feses berdarah": "bloody_stool", "darah di tinja": "bloody_stool",
    "kulit kering": "drying_and_tingling_lips", "bibir kering": "drying_and_tingling_lips",
    "rambut rontok": "hair_fall", "jerawat": "pus_filled_pimples",
    "kulit berminyak": "oily_skin", "kulit mengelupas": "scurring",
    "memar": "bruising", "luka sulit sembuh": "obesity",
    "pembengkakan": "swelling_joints", "sendi bengkak": "swelling_joints",
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


def extract_symptoms_from_text(message: str, feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Ekstrak gejala dari kalimat bebas Bahasa Indonesia/Inggris.

    Mendukung frasa multi-kata, alias, typo ringan, dan negasi sederhana.
    """
    text = _clean_text(message)
    if not text:
        return [], []

    tokens = text.split()
    negations = {"tidak", "nggak", "enggak", "gak", "ga", "bukan", "tanpa", "belum"}

    def is_negated(position: int) -> bool:
        return any(token in negations for token in tokens[max(0, position - 3):position])

    # Cari frasa terpanjang lebih dulu supaya "sakit kepala" tidak diproses
    # sebagai dua kata yang terpisah.
    phrases = set(SYMPTOM_ALIASES)
    phrases.update(_feature_to_text(feature) for feature in feature_cols)
    ordered_phrases = sorted(phrases, key=lambda item: len(item.split()), reverse=True)

    found = {}
    for phrase in ordered_phrases:
        phrase_tokens = phrase.split()
        size = len(phrase_tokens)
        if not size:
            continue

        for i in range(len(tokens) - size + 1):
            if tokens[i:i + size] != phrase_tokens:
                continue
            if is_negated(i):
                continue

            feature = normalize_symptom(phrase, feature_cols)
            if feature:
                found[feature] = phrase

    # Fuzzy matching untuk typo satu kata. Ambang cukup tinggi agar
    # kata biasa tidak mudah dianggap sebagai gejala.
    feature_lookup = {
        _feature_to_text(feature): feature
        for feature in feature_cols
    }
    for i, token in enumerate(tokens):
        if len(token) < 5 or is_negated(i):
            continue

        close = get_close_matches(
            token,
            list(feature_lookup.keys()),
            n=1,
            cutoff=0.90,
        )
        if close:
            found[feature_lookup[close[0]]] = token
            continue

        alias_close = get_close_matches(
            token,
            list(SYMPTOM_ALIASES.keys()),
            n=1,
            cutoff=0.92,
        )
        if alias_close:
            alias_feature = SYMPTOM_ALIASES[alias_close[0]]
            if alias_feature in feature_cols:
                found[alias_feature] = token

    return sorted(found), []
