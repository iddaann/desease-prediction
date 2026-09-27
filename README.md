# HealthPredict — Chatbot Prediksi Penyakit

HealthPredict adalah aplikasi web untuk membantu pengguna umum memahami keluhan kesehatan berdasarkan gejala yang mereka masukkan. Pengguna berinteraksi melalui antarmuka chatbot; backend menormalisasi gejala, menjalankan model machine learning, lalu menampilkan beberapa kemungkinan penyakit, confidence score, dan rekomendasi umum.

> **Catatan penting:** HealthPredict adalah alat bantu informasi/screening berbasis model, bukan alat diagnosis. Hasil model tidak boleh digunakan sebagai pengganti pemeriksaan tenaga kesehatan.

## Fitur
- Chatbot gejala berbasis web responsif.
- Input keluhan dalam bahasa natural sederhana.
- Normalisasi alias gejala Bahasa Indonesia → fitur dataset.
- Prediksi penyakit menggunakan Random Forest.
- Top-5 kandidat penyakit + probabilitas.
- Confidence score.
- Rekomendasi tindak lanjut umum dan peringatan kondisi darurat.
- Riwayat konsultasi per browser/session.
- Penyimpanan data konsultasi dengan opsi enkripsi Fernet.
- Login administrator.
- Dashboard admin.
- Evaluasi accuracy, precision, recall, F1 dan cross-validation.
- Retrain model dari data/Training.csv.
- Health check endpoint.
- Docker image yang melatih model ketika image dibangun.

## Struktur
```text
api/
  auth.py
  chat_schemas.py
  db.py
  disease_ml.py
  main.py
src/
  preprocessing.py
  symptom_normalizer.py
  train.py
  evaluate.py
  evaluate_cv.py
data/
  Training.csv
  Testing.csv
frontend/
  index.html
models/
  disease/          # dibuat saat training/build Docker
DockerFile
requirements.txt
```

## Dataset
Model menggunakan data/Training.csv dengan pola:
- kolom fitur = gejala biner
- kolom target = prognosis

Jalankan training:
```powershell
python -m src.train
```
Evaluasi pada testing set:
```powershell
python src/evaluate.py
```
Cross-validation:
```powershell
python src/evaluate_cv.py
```

## Menjalankan lokal
```powershell
pip install -r requirements.txt
python -m src.train
python -m uvicorn api.main:app --reload
```
Buka: http://127.0.0.1:8000/

Frontend sekarang dilayani langsung oleh FastAPI, sehingga tidak perlu lagi membuka frontend/index.html lewat Live Server.

## Environment
Contoh:
```text
ADMIN_NAME=Administrator
ADMIN_EMAIL=admin@healthbot.local
ADMIN_PASSWORD=ganti-password-kuat
DATA_ENCRYPTION_KEY=<Fernet key>
DATABASE_PATH=data/app.db
CORS_ORIGINS=https://domain-frontend.example
```
Generate Fernet key:
```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
Untuk deployment, gunakan secret manager/environment variables dan jangan commit file .env.

Jika DATA_ENCRYPTION_KEY tidak diset, mode development menyimpan field konsultasi dengan prefix plain:. Untuk production, **wajib** mengatur DATA_ENCRYPTION_KEY.

## Akun administrator development
Default:
```text
Email    : admin@healthbot.local
Password : admin123
```
Untuk deployment, ubah melalui ADMIN_EMAIL dan ADMIN_PASSWORD sebelum database pertama kali dibuat.

## API
GET /health
POST /api/chat
GET /api/history/{session_id}
GET /api/symptoms
POST /auth/login
GET /admin/dashboard
GET /admin/model-metrics
GET /admin/consultations
POST /admin/retrain

## Docker
Build:
```powershell
docker build -f DockerFile -t healthpredict .
```
Run:
```powershell
docker run --rm -p 8000:8000 -e ADMIN_EMAIL=admin@example.com -e ADMIN_PASSWORD="ganti-password-kuat" -e DATA_ENCRYPTION_KEY="PASTE_FERNET_KEY" healthpredict
```
Open http://localhost:8000

Docker image menyalin dataset dan menjalankan training saat build, sehingga model tersedia sebelum container menerima request.

## Production deployment

The repository includes `render.yaml` for a production deployment on Render:
- FastAPI + frontend run as one Docker web service.
- Render PostgreSQL is used through `DATABASE_URL`.
- Render provides managed TLS/HTTPS at the edge.
- `/health` is configured as the deployment health check.
- Production secrets are requested through Blueprint `sync: false` variables.

Render automatically provisions and renews TLS certificates and redirects HTTP to HTTPS. The application itself therefore continues to serve plain HTTP inside the Render service.

Before the first production deploy, provide:
- `ADMIN_EMAIL`
- `ADMIN_PASSWORD`
- `DATA_ENCRYPTION_KEY`

Do not commit these values.

### Production runtime smoke test

After the service is deployed, run:

```powershell
python scripts/smoke_test.py https://YOUR-SERVICE.onrender.com
```

The script checks:
1. `/health`
2. `/api/symptoms`
3. `POST /api/chat`

For local runtime testing:

```powershell
python scripts/smoke_test.py http://127.0.0.1:8000
```

### Logging and monitoring

The API writes request timing, status codes, request IDs, startup messages, health-check failures, and unhandled exceptions to stdout/stderr. It intentionally does not log chat message bodies, symptoms, or prediction payloads in request logs to reduce exposure of sensitive health information.

Each request receives an `X-Request-ID` response header. If a user reports an error, that ID can be matched with the deployment logs.

The `/health` endpoint returns HTTP 503 when the database or model is unavailable, allowing the hosting platform to detect an unhealthy instance.

## Deployment checklist
1. Gunakan HTTPS pada reverse proxy/platform deployment.
2. Ganti akun administrator default melalui environment variables sebelum database dibuat.
3. Set DATA_ENCRYPTION_KEY.
4. Set CORS_ORIGINS hanya ke origin frontend yang diperlukan.
5. Gunakan persistent storage untuk data/app.db jika memakai SQLite.
6. Untuk skala besar, migrasikan database ke PostgreSQL.
7. Jangan mengunggah model/data dari sumber yang tidak dipercaya.
8. Uji model menggunakan dataset yang representatif sebelum penggunaan nyata.
9. Monitor confidence rendah dan error model secara berkala.

## Catatan SRS
Implementasi ini mempertahankan bagian SRS yang masih relevan untuk konsep chatbot: preprocessing, klasifikasi, confidence score, rekomendasi, penyimpanan riwayat, evaluasi model, retraining, keamanan, dan antarmuka web responsif.
Bagian SRS yang khusus untuk lingkungan rumah sakit seperti estimasi lama rawat inap dan billing tidak digunakan karena konsep produk yang diimplementasikan adalah chatbot prediksi penyakit berbasis gejala.

## Evaluasi ML dan batasan penggunaan

HealthPredict diposisikan sebagai **project pembelajaran machine learning**, bukan alat diagnosis medis. Model saat ini mempelajari dataset gejala-penyakit publik dan hasil evaluasinya hanya menggambarkan performa pada dataset tersebut. Skor model pada prediksi bukan probabilitas diagnosis yang terkalibrasi.

Sebelum eksperimen model, audit dataset dapat dijalankan:

```powershell
python scripts/dataset_audit.py
```

Laporan tersimpan di `reports/dataset_audit.json` dan memeriksa ukuran dataset, distribusi kelas, duplicate, missing value, jumlah gejala per baris, feature non-biner, perbedaan kelas train/test, dan exact feature-vector overlap.

Untuk membandingkan baseline classifier:

```powershell
python scripts/model_compare.py
```

Laporan tersimpan di `reports/model_comparison.json`. Perbandingan menggunakan accuracy dan macro precision/recall/F1 serta cross-validation macro F1; jangan memilih model hanya dari accuracy.

Evaluasi training model juga menyimpan:
- holdout accuracy, weighted/macro precision, recall, dan F1;
- confusion matrix dan classification report per kelas;
- cross-validation accuracy dan macro F1;
- evaluasi tambahan terhadap `data/Testing.csv` jika tersedia.

Input dengan tanda peringatan seperti sesak berat, nyeri dada berat, pingsan, kejang, atau penurunan kesadaran memicu safety alert sebelum hasil model ditafsirkan. Safety layer ini bukan diagnosis; tujuannya mencegah output model menjadi satu-satunya dasar ketika terdapat tanda yang membutuhkan perhatian segera.

### Batasan dataset

Dataset publik/sederhana dapat berbeda jauh dari data klinis nyata. Karena itu, metric tinggi pada dataset ini **tidak boleh ditafsirkan sebagai validasi klinis atau bukti bahwa model akurat untuk masyarakat umum**. Project ini tidak melakukan clinical validation.


### Temuan audit dataset saat ini

Audit terhadap file yang ada di repository menunjukkan beberapa hal penting:

- `data/Training.csv`: 4.920 baris, 132 feature gejala, dan 41 kelas penyakit.
- Distribusi kelas training terlihat seimbang: 120 baris per kelas sebelum duplicate dihapus.
- Setelah melihat kombinasi feature gejala, hanya 304 kombinasi feature yang unik; 4.616 baris merupakan duplikasi kombinasi feature yang sama dengan target yang sama.
- `data/Testing.csv`: 42 baris dan 41 kelas.
- 41 dari 42 kombinasi feature pada testing memiliki kecocokan persis dengan kombinasi feature yang sudah ada di training.
- Karena itu, `Testing.csv` **tidak layak dianggap sebagai independent external validation set** untuk project ini. Model tetap boleh menggunakannya sebagai evaluasi eksploratif, tetapi hasilnya harus diberi catatan keterbatasan.

Implikasinya, project sekarang sengaja melakukan `drop_duplicates()` sebelum training dan menambahkan evaluasi holdout serta cross-validation. Langkah berikutnya yang lebih kuat adalah mencari dataset yang benar-benar independen jika ingin menguji generalisasi model.
