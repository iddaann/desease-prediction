# Disease Prediction (Edukasi)

Aplikasi eksperimen machine learning untuk memetakan daftar gejala ke beberapa label penyakit menggunakan Random Forest. **Aplikasi ini bukan alat diagnosis, bukan pengganti tenaga kesehatan, dan belum divalidasi untuk penggunaan klinis.**

Dataset yang digunakan berasal dari [Disease Prediction Using Machine Learning di Kaggle](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning). Dataset publik yang terstruktur dapat memiliki kombinasi gejala berulang dan pola label yang terlalu sederhana dibandingkan kondisi pasien nyata.

## Struktur proyek

```
api/                 FastAPI API dan skema request/response
data/                Training.csv dan Testing.csv
frontend/index.html  UI chat sederhana
models/              Artefak model yang telah dilatih
notebooks/            Analisis dan eksperimen
src/
  preprocessing.py   Load/bersihkan data
  train.py           Latih model
  evaluate.py        Evaluasi pada data testing
  evaluate_cv.py     Cross-validation dengan grouping kombinasi gejala
  pewdict.py         Prediksi dari daftar gejala
  symptom_normalizer.py Normalisasi gejala
```

## Menjalankan lokal

Disarankan Python 3.11 atau 3.12 untuk menyesuaikan image Docker.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Latih ulang model dari dataset yang tersedia
python src/train.py

# Evaluasi data testing terpisah
python src/evaluate.py

# Evaluasi silang dengan grouping kombinasi gejala
python src/evaluate_cv.py

# Jalankan API
uvicorn api.main:app --reload
```

API docs tersedia di `http://127.0.0.1:8000/docs`; health check di `/health`.

## Frontend

Buka `frontend/index.html` melalui server lokal, misalnya VS Code Live Server. Frontend memakai `API_URL` yang bisa diatur melalui konfigurasi sebelum deploy. Jangan gunakan URL loopback (`127.0.0.1`) untuk backend yang di-hosting karena alamat tersebut menunjuk ke perangkat pengguna.

Untuk deployment, set environment variable `CORS_ORIGINS` ke daftar origin frontend yang diizinkan, dipisahkan koma, contoh:

```text
CORS_ORIGINS=https://aplikasi-anda.example
```

Jangan gunakan wildcard CORS untuk aplikasi produksi. Pastikan URL backend memakai HTTPS jika frontend juga HTTPS.

## Request API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms":["itching","skin_rash"]}'
```

## Interpretasi hasil

- Nilai dari `predict_proba` adalah skor keluaran model, **bukan probabilitas klinis yang terkalibrasi**.
- Skor tidak boleh ditampilkan atau dipahami sebagai kepastian seseorang menderita penyakit tertentu.
- Gejala yang tidak dikenali atau input yang belum cukup harus diperbaiki/ditanyakan; model tidak seharusnya memberi label penyakit seolah pasti.
- Nyeri dada berat, kesulitan bernapas berat, penurunan kesadaran, atau gejala gawat lain memerlukan pertolongan medis segera. Jangan menunggu keluaran aplikasi.
- Dataset dan metrik saat ini tidak membuktikan akurasi pada pasien nyata. Sebelum penggunaan klinis diperlukan data representatif, validasi eksternal, kalibrasi, tinjauan tenaga kesehatan, pengujian bias, dan penilaian risiko.

## Evaluasi yang disarankan

Jangan hanya melaporkan accuracy. Lihat macro-F1, weighted-F1, laporan per kelas, confusion matrix, kalibrasi, serta hasil pada data eksternal yang tidak digunakan saat training. Cross-validation pada script ini mengelompokkan kombinasi gejala identik agar kombinasi yang sama tidak tersebar di train dan validation; hal itu membantu mengurangi leakage, tetapi **tidak** menggantikan validasi eksternal.

## Batasan

Proyek ini adalah prototipe pembelajaran. Jangan menggunakannya untuk diagnosis mandiri, menentukan obat, menunda konsultasi, atau mengambil keputusan medis. Evaluasi kode yang baik sekalipun tidak cukup untuk menyatakan sistem aman secara klinis.
