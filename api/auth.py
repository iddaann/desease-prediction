import secrets
from datetime import datetime, timedelta, timezone
from fastapi import Depends, Header, HTTPException
from api.db import connect

def create_session(user_id):
    token = secrets.token_urlsafe(48)
    expires = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    with connect() as db:
        db.execute("INSERT INTO sessions(token,user_id,expires_at) VALUES(?,?,?)", (token, user_id, expires))
    return token

def get_current_user(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Autentikasi diperlukan")
    token = authorization.removeprefix("Bearer ").strip()
    with connect() as db:
        row = db.execute("SELECT u.id,u.name,u.email,u.role,s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?", (token,)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Token tidak valid")
    if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Sesi telah berakhir")
    return dict(row)

def require_admin(user=Depends(get_current_user)):
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Akses administrator diperlukan")
    return user
