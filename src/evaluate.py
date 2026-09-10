"""
Script evaluasi: load model yang sudah dilatih, uji ke Testing.csv,
tampilkan akurasi + classification report + top feature importance.
"""
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def evaluate():
    _, _, X_test, y_test, feature_cols = load_dataset()

    model = joblib.load(MODELS_DIR / "desease_model.joblib")
    encoder = joblib.load(MODELS_DIR / "label_encoder.joblib")

    y_test_enc = encoder.transform(y_test)
    y_pred_enc = model.predict(X_test)

    acc = accuracy_score(y_test_enc, y_pred_enc)
    print(f"Akurasi di data testing: {acc:.4f} ({acc*100:.2f}%)")
    print()
    print(classification_report(
        y_test_enc, y_pred_enc,
        target_names=encoder.classes_,
        zero_division=0,
    ))

    # Baris yang salah diprediksi (kalau ada) -- berguna untuk debugging
    mismatches = X_test[y_pred_enc != y_test_enc]
    if len(mismatches) > 0:
        print(f"\n{len(mismatches)} baris salah prediksi:")
        for idx in mismatches.index:
            asli = encoder.inverse_transform([y_test_enc[X_test.index.get_loc(idx)]])[0]
            prediksi = encoder.inverse_transform([y_pred_enc[X_test.index.get_loc(idx)]])[0]
            print(f"  - Asli: {asli} | Prediksi: {prediksi}")

    importances = model.feature_importances_
    top = sorted(zip(feature_cols, importances), key=lambda x: -x[1])[:15]
    print("\nTop 15 gejala paling berpengaruh:")
    for gejala, skor in top:
        print(f"  {gejala}: {skor:.4f}")

    return acc


if __name__ == "__main__":
    evaluate()
