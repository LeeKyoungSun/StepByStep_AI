from fastapi import Header, HTTPException
from app.schemas.common import User

def get_current_user(Authorization: str = Header(..., alias="Authorization")) -> User:
    """
    데모용 간단 파서: "Bearer <userId>:<email>"
    실제 서비스에서는 PyJWT 등으로 검증하세요.
    """
    if not Authorization.startswith("Bearer "):
        raise HTTPException(401, "UNAUTHORIZED")
    raw = Authorization.split(" ", 1)[1].strip()
    try:
        uid, email = raw.split(":", 1)
        return User(userId=int(uid), email=email)
    except Exception:
        raise HTTPException(401, "UNAUTHORIZED")