import hashlib
import hmac
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "data" / "app.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

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

def seed_demo_users():
    """
    Create the demo accounts shown by the frontend when they do not exist yet.
    Existing users are not overwritten.
    """
    demo_users = [
        ("Tenaga Medis Demo", "medis@healthbot.local", "medis123", "medical"),
        ("Administrator Demo", "admin@healthbot.local", "admin123", "admin"),
    ]
    with connect() as db:
        for name, email, password, role in demo_users:
            db.execute(
                """
                INSERT OR IGNORE INTO users(name, email, password_hash, role, created_at)
                VALUES(?,?,?,?,?)
                """,
                (name, email, hash_password(password), role, utcnow()),
            )

def init_db():
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
