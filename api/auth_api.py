from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import jwt
from datetime import datetime, timedelta
from common.config import Config

router = APIRouter()

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=Config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, Config.JWT_SECRET_KEY, algorithm=Config.JWT_ALGORITHM)
    return encoded_jwt

@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    # Simple check for default admin credentials
    if request.username == Config.DEFAULT_USERNAME and request.password == Config.DEFAULT_PASSWORD:
        access_token = create_access_token(
            data={"sub": request.username}
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(
        status_code=401,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    ) 