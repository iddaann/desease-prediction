# Disease Prediction

Prediksi penyakit dari checklist gejala, menggunakan Random Forest.
Dataset: [Disease Prediction Using Machine Learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) (132 gejala biner -> 41 penyakit).

## Struktur

```
data/          Dataset mentah (Training.csv, Testing.csv)
src/
  preprocessing.py   Load & bersihkan data
  train.py           Latih model, simpan ke models/
  evaluate.py         Uji model ke data testing
  pewdict.py         Fungsi prediksi dari daftar gejala
models/        Artefak hasil training (.joblib)
api/
  main.py       FastAPI app, endpoint /predict
  schemas.py    Skema request/response
notebooks/      (belum diisi -- untuk eksplorasi & dokumentasi analisis)
```

## Cara Pakai

```bash
pip install -r requirements.txt

# Latih model (hasil disimpan ke models/)
python src/train.py

# Evaluasi ke data testing
python src/evaluate.py

# Jalankan API
uvicorn api.main:app --reload
```

Contoh request ke API:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["itching", "skin_rash", "high_fever"]}'
```

## Performa Model

Random Forest (200 trees) mencapai **97.62% akurasi** di data testing (41/42 benar).
Catatan: dataset ini bersifat deterministik (tiap penyakit punya kombinasi gejala tetap),
jadi akurasi tinggi ini wajar dan bukan indikasi model akan seakurat ini pada gejala
pasien sungguhan yang lebih beragam/ambigu.
