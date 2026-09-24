import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "data" / "app.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def using_postgres():
    return bool(DATABASE_URL)

def utcnow():
    return datetime.now(timezone.utc).isoformat()

class SQLiteConnection:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def execute(self, sql, params=()):
        return self.conn.execute(sql, params)

    def executescript(self, script):
        return self.conn.executescript(script)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.conn.close()

    def close(self):
        self.conn.close()

class PostgresConnection:
    def __init__(self, url):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError("psycopg belum terpasang. Jalankan: pip install psycopg[binary]") from exc
        self.conn = psycopg.connect(url, row_factory=dict_row)

    def execute(self, sql, params=()):
        return self.conn.execute(sql.replace("?", "%s"), params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type:
            self.conn.rollback()
        else:
            self.conn.commit()
        self.conn.close()

    def close(self):
        self.conn.close()

def connect():
    return PostgresConnection(DATABASE_URL) if using_postgres() else SQLiteConnection(DB_PATH)

def hash_password(password, salt=None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 120000)
    return salt.hex() + "$" + digest.hex()

def verify_password(password, encoded):
    try:
        salt_hex, digest_hex = encoded.split("$", 1)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 120000)
        return hmac.compare_digest(actual, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False

def _fernet():
    key = os.getenv("DATA_ENCRYPTION_KEY")
    if not key or Fernet is None:
        return None
    return Fernet(key.encode())

def encrypt_text(value):
    value = str(value)
    cipher = _fernet()
    if cipher is None:
        if os.getenv("ENVIRONMENT", "development").lower() == "production":
            raise RuntimeError("DATA_ENCRYPTION_KEY wajib diisi pada production.")
        return "plain:" + value
    return "fernet:" + cipher.encrypt(value.encode()).decode()

def decrypt_text(value):
    if value is None:
        return ""
    value = str(value)
    if value.startswith("fernet:"):
        cipher = _fernet()
        if cipher is None:
            raise RuntimeError("DATA_ENCRYPTION_KEY diperlukan untuk membaca data terenkripsi.")
        return cipher.decrypt(value[7:].encode()).decode()
    return value[6:] if value.startswith("plain:") else value

def generate_encryption_key():
    if Fernet is None:
        raise RuntimeError("Paket cryptography belum terpasang.")
    return Fernet.generate_key().decode()

def seed_demo_users():
    production = os.getenv("ENVIRONMENT", "development").lower() == "production"
    name = os.getenv("ADMIN_NAME", "Administrator")
    email = os.getenv("ADMIN_EMAIL", "").strip().lower() if production else os.getenv("ADMIN_EMAIL", "admin@healthbot.local").strip().lower()
    password = os.getenv("ADMIN_PASSWORD", "") if production else os.getenv("ADMIN_PASSWORD", "admin123")
    if production and (not email or not password or password == "admin123"):
        raise RuntimeError("ADMIN_EMAIL dan ADMIN_PASSWORD production wajib diatur dan tidak boleh menggunakan default.")
    role = "admin"
    with connect() as db:
        db.execute(
            "INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(email) DO NOTHING",
            (name, email, hash_password(password), role, utcnow()),
        )

def init_db():
    if using_postgres():
        statements = [
            """CREATE TABLE IF NOT EXISTS users (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('medical','admin')),
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(id),
                patient_payload TEXT NOT NULL,
                result_payload TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS consultations (
                id BIGSERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                message TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                result TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_consultations_session ON consultations(session_id)",
            """CREATE TABLE IF NOT EXISTS model_runs (
                id BIGSERIAL PRIMARY KEY,
                model_version TEXT NOT NULL,
                status TEXT NOT NULL,
                metrics_payload TEXT NOT NULL,
                dataset_name TEXT,
                created_at TEXT NOT NULL
            )""",
        ]
        with connect() as db:
            for statement in statements:
                db.execute(statement)
    else:
        with connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('medical','admin')),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                patient_payload TEXT NOT NULL,
                result_payload TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                result TEXT NOT NULL,
                model_version TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_consultations_session ON consultations(session_id);
            CREATE TABLE IF NOT EXISTS model_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                status TEXT NOT NULL,
                metrics_payload TEXT NOT NULL,
                dataset_name TEXT,
                created_at TEXT NOT NULL
            );
            """)
    seed_demo_users()

def database_health():
    with connect() as db:
        row = db.execute("SELECT 1 AS ok").fetchone()
    return bool(row["ok"] if isinstance(row, dict) else row["ok"])
