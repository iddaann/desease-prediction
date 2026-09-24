# Disease Prediction & Health Recommendation

Implementasi SRS **Sistem Prediksi Penyakit dan Rekomendasi Kesehatan Berbasis Machine Learning**.

## Fitur yang sudah diimplementasikan

- Login dan autentikasi berbasis token.
- Role **Tenaga Medis** dan **Administrator**.
- Form input data pasien dan validasi.
- REST API JSON.
- Penyimpanan riwayat input dan hasil prediksi di SQLite.
- Model klasifikasi Random Forest.
- Model regresi Random Forest untuk lama rawat inap.
- Model regresi Random Forest untuk estimasi biaya.
- Confidence score dan tiga kandidat klasifikasi.
- Rekomendasi tindak lanjut berbasis hasil prediksi.
- Riwayat prediksi pengguna.
- Dashboard administrator.
- Evaluasi akurasi, precision, recall, F1, RMSE, MAE, dan R2.
- Retrain model dari dataset terbaru.
- Versioning model dan pencatatan proses retrain.
- Frontend web responsif tanpa instalasi tambahan.
- Model symptom-disease lama tetap disimpan di `src/` sebagai modul terpisah.

## Catatan penting tentang dataset

SRS meminta dataset dengan data administratif/klinis dan target **Medical Condition**, **Length of Stay**, serta **Billing Amount**. Repository awal justru menggunakan dataset gejala biner dengan target `prognosis`. Karena kedua dataset mempunyai struktur dan target yang berbeda, model lama tidak boleh dipakai untuk mengarang estimasi lama rawat atau biaya.

Repository ini karena itu menyediakan pipeline SRS yang akan aktif setelah dataset rumah sakit tersedia. Gunakan `data/hospital_training.example.csv` sebagai contoh struktur. Dataset nyata yang dipakai untuk produksi harus divalidasi dan memiliki minimal kolom:

```
Age
Gender
Blood Type
Admission Type
Medication
Test Results
Medical Condition
Length of Stay
Billing Amount
```

## Menjalankan

```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
```

Buka `frontend/index.html` melalui browser.

Akun demo:
- Tenaga medis: `medis@healthbot.local` / `medis123`
- Administrator: `admin@healthbot.local` / `admin123`

Untuk model SRS rumah sakit, tempatkan dataset pada `data/hospital_training.csv`, lalu login sebagai administrator dan jalankan **Retrain Model**.

## Endpoint utama

- `POST /auth/login`
- `GET /auth/me`
- `POST /predict`
- `GET /predictions`
- `GET /admin/dashboard`
- `GET /admin/model-metrics`
- `POST /admin/retrain`

## Batasan penggunaan

Sistem adalah alat bantu skrining dan perencanaan. Hasil model bukan diagnosis resmi. Akurasi sangat bergantung pada kualitas, representativitas, dan validasi dataset klinis. Implementasi produksi perlu HTTPS, database server seperti PostgreSQL/MySQL, secret management, audit log yang lebih lengkap, dan pengamanan jaringan.

