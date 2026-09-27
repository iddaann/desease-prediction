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

## Database dan Environment

Aplikasi **sudah memiliki kode inisialisasi database**, tetapi file database tidak disimpan di repository.

Saat berjalan di development tanpa `DATABASE_URL`, aplikasi menggunakan SQLite pada:
```text
data/app.db
```

Tabel akan dibuat otomatis ketika backend dijalankan karena `api.main` memanggil `init_db()` saat startup. Jadi tidak perlu membuat tabel SQLite secara manual.

Untuk production, aplikasi menggunakan PostgreSQL jika `DATABASE_URL` tersedia. Repository juga menyediakan `render.yaml` yang mendefinisikan database PostgreSQL untuk deployment Render. File konfigurasi tersebut **belum berarti database production sudah dibuat atau sudah terhubung**; database production baru tersedia setelah service/deployment Render benar-benar dibuat dan environment variable `DATABASE_URL` diberikan.

Contoh environment development:
```text
ENVIRONMENT=development
ADMIN_NAME=Administrator
ADMIN_EMAIL=admin@healthbot.local
ADMIN_PASSWORD=ganti-password-kuat
DATA_ENCRYPTION_KEY=<Fernet key>
DATABASE_PATH=data/app.db
CORS_ORIGINS=
LOG_LEVEL=INFO
```
Generate Fernet key:
```powershell
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
Untuk deployment, gunakan secret manager/environment variables dan jangan commit file .env.

Jika DATA_ENCRYPTION_KEY tidak diset, mode development menyimpan field konsultasi dengan prefix plain:. Untuk production, **wajib** mengatur DATA_ENCRYPTION_KEY.

## Akun administrator development

Pada mode development, aplikasi menggunakan nilai default berikut **hanya jika environment variable belum diatur**:
```text
Email    : admin@healthbot.local
Password : gunakan nilai ADMIN_PASSWORD milik environment lokal
```

Sebaiknya tetap mengatur `ADMIN_EMAIL` dan `ADMIN_PASSWORD` sendiri pada environment lokal. Untuk production, keduanya **wajib** diisi dan tidak boleh menggunakan password default.

Perlu diperhatikan bahwa `seed_demo_users()` hanya membuat akun administrator ketika email tersebut belum ada di database. Mengubah `ADMIN_PASSWORD` setelah akun sudah tersimpan tidak otomatis mengganti password hash yang sudah ada.

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
2. Set `ADMIN_EMAIL` dan `ADMIN_PASSWORD` production sebelum database pertama kali diinisialisasi.
3. Set `DATA_ENCRYPTION_KEY`.
4. Set `CORS_ORIGINS` hanya ke origin frontend yang diperlukan.
5. Untuk SQLite, gunakan persistent storage karena `data/app.db` adalah file lokal.
6. Untuk production yang menggunakan Render, pastikan PostgreSQL service benar-benar sudah dibuat dan `DATABASE_URL` terisi.
7. Jangan mengunggah model/data dari sumber yang tidak dipercaya.
8. Uji model menggunakan dataset yang representatif sebelum penggunaan nyata.
9. Monitor confidence rendah dan error model secara berkala.

### Status database saat ini

Repository ini sudah menyediakan **skema dan inisialisasi database**, tetapi tidak menyimpan database production di Git.

- Development: SQLite `data/app.db` dibuat otomatis saat aplikasi pertama kali dijalankan.
- Production: PostgreSQL dikonfigurasi melalui `DATABASE_URL`.
- Render: `render.yaml` mendefinisikan resource PostgreSQL, tetapi resource tersebut baru benar-benar tersedia setelah deployment/provisioning Render dilakukan.

Jadi, untuk kondisi repository saat ini, yang sudah dibuat adalah **kode database dan skemanya**, bukan database production yang sudah online.

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

### Eksperimen robustness input gejala

Project juga menyediakan `scripts/robustness_test.py` untuk menguji perilaku model ketika informasi gejala tidak lengkap. Eksperimen menggunakan pola gejala unik setelah deduplikasi, mengambil holdout, lalu menghapus secara sintetis 1, 2, dan 3 gejala positif dari setiap input. Hasil disimpan di `reports/robustness_report.json`.

Eksperimen ini ditujukan untuk pembelajaran dan pengujian robustness chatbot, bukan simulasi pasien atau validasi klinis. Penghapusan gejala secara sintetis tidak mewakili cara pasien sebenarnya mendeskripsikan keluhan.



### Ringkasan hasil evaluasi model saat ini

Eksperimen dilakukan pada pola gejala unik setelah deduplikasi. Data dibagi menjadi training dan holdout secara stratified dengan random state 42. Karena jumlah minimum sampel per kelas pada training split adalah 4, cross-validation menggunakan 3 fold.

Hasil perbandingan model:

| Model | Holdout Accuracy | Holdout Macro F1 | CV Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| SVM RBF | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| Random Forest | 1.0000 | 1.0000 | 0.9946 ± 0.0077 |

Hasil tersebut menunjukkan bahwa ketiga baseline dapat memisahkan kelas dengan sangat baik pada dataset yang tersedia. Namun, angka 1.0000 tidak boleh dianggap sebagai performa klinis atau performa yang pasti terjadi pada data pengguna nyata. Struktur dataset yang sangat teratur dan tingginya duplikasi pola gejala menjadi keterbatasan penting.

Random Forest juga diuji dengan holdout yang benar-benar tidak digunakan saat fitting model robustness. Hasil pengurangan informasi gejala secara sintetis:

| Kondisi input | Accuracy | Macro F1 |
|---|---:|---:|
| Gejala lengkap | 1.0000 | 1.0000 |
| Dikurangi 1 gejala | 0.9344 | 0.9024 |
| Dikurangi 2 gejala | 0.9508 | 0.9220 |
| Dikurangi 3 gejala | 0.8689 | 0.8070 |

Nilai tersebut menunjukkan bahwa performa model berubah ketika informasi gejala dikurangi. Hasil "dikurangi 2 gejala" yang sedikit lebih tinggi daripada "dikurangi 1 gejala" tidak diartikan bahwa lebih sedikit gejala selalu menghasilkan prediksi lebih baik; penghapusan dilakukan secara sintetis sehingga variasi kombinasi gejala dapat menghasilkan perbedaan tersebut.

### Feature importance Random Forest

Eksperimen feature importance dilakukan menggunakan Random Forest yang hanya dilatih pada training split. Lima belas fitur dengan importance tertinggi pada eksperimen saat ini adalah:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | vomiting | 0.019036 |
| 2 | fatigue | 0.017979 |
| 3 | muscle_pain | 0.017686 |
| 4 | sweating | 0.016620 |
| 5 | diarrhoea | 0.015986 |
| 6 | itching | 0.015144 |
| 7 | extra_marital_contacts | 0.014792 |
| 8 | headache | 0.014757 |
| 9 | weight_loss | 0.014023 |
| 10 | abdominal_pain | 0.013660 |
| 11 | joint_pain | 0.013538 |
| 12 | pus_filled_pimples | 0.013529 |
| 13 | high_fever | 0.013416 |
| 14 | bladder_discomfort | 0.013376 |
| 15 | yellowing_of_eyes | 0.013341 |

Feature importance ini hanya menjelaskan kontribusi relatif fitur terhadap keputusan Random Forest pada dataset eksperimen. Nilai tersebut bukan ukuran tingkat kepentingan medis suatu gejala dan bukan bukti hubungan sebab-akibat.

### Kesimpulan evaluasi

Berdasarkan eksperimen yang sudah dilakukan, pipeline ML saat ini telah mencakup:
- audit kualitas dan duplikasi dataset;
- stratified holdout evaluation;
- perbandingan Logistic Regression, SVM RBF, dan Random Forest;
- cross-validation macro F1;
- evaluasi robustness ketika sebagian gejala dihilangkan;
- analisis feature importance Random Forest;
- pengecekan keterbatasan `Testing.csv` sebagai validation set.

Hasil evaluasi cukup untuk mendokumentasikan project ini sebagai eksperimen pembelajaran machine learning end-to-end. Namun, hasil tersebut belum cukup untuk menyatakan bahwa model tervalidasi untuk diagnosis atau penggunaan klinis. Validasi yang lebih kuat memerlukan data independen dan representatif serta validasi medis yang berada di luar cakupan project pembelajaran ini.

### Analisis perilaku model

`scripts/model_compare.py` juga menyimpan 15 fitur dengan nilai feature importance tertinggi dari Random Forest ke `reports/model_comparison.json`. Feature importance digunakan untuk mempelajari perilaku model terhadap fitur gejala, bukan untuk menyimpulkan hubungan medis atau penyebab penyakit.
