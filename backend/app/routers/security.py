from fastapi import APIRouter
from pydantic import BaseModel
from ..services.security_service import security_service

router = APIRouter(prefix="/security", tags=["security"])

class EncryptRequest(BaseModel):
    data: str

class EncryptResponse(BaseModel):
    token: str

class DecryptRequest(BaseModel):
    token: str

class DecryptResponse(BaseModel):
    data: str

@router.post('/encrypt', response_model=EncryptResponse)
async def encrypt(req: EncryptRequest):
    return EncryptResponse(token=security_service.encrypt(req.data))

@router.post('/decrypt', response_model=DecryptResponse)
async def decrypt(req: DecryptRequest):
    return DecryptResponse(data=security_service.decrypt(req.token))
