"""
Modul preprocessing: load & bersihkan dataset gejala-penyakit.
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_raw(train_path: Path = None, test_path: Path = None):
    """Load Training.csv dan Testing.csv mentah."""
    train_path = train_path or DATA_DIR / "Training.csv"
    test_path = test_path or DATA_DIR / "Testing.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Buang kolom sampah (mis. 'Unnamed: 133' yang kosong)."""
    junk_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    return df.drop(columns=junk_cols)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Semua kolom gejala (biner), tanpa kolom target 'prognosis'."""
    return [c for c in df.columns if c != "prognosis"]


def load_dataset(train_path: Path = None, test_path: Path = None):
    """
    Pipeline lengkap: load -> clean -> split X/y.
    Return: X_train, y_train, X_test, y_test, feature_cols
    """
    train, test = load_raw(train_path, test_path)
    train = clean(train)
    test = clean(test)

    feature_cols = get_feature_columns(train)
    X_train, y_train = train[feature_cols], train["prognosis"]
    X_test, y_test = test[feature_cols], test["prognosis"]

    return X_train, y_train, X_test, y_test, feature_cols
