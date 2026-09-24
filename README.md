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