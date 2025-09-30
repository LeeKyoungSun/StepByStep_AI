from pydantic import BaseModel

class User(BaseModel):
    userId: int
    email: str

class ErrorResponse(BaseModel):
    error: dict  # {"code": "...", "message": "..."}