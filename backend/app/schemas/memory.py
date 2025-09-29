from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class MemoryCreate(BaseModel):
    user_id: Optional[int] = None
    text: str = Field(..., max_length=1000)
    metadata: Optional[Dict[str, Any]] = None

class MemoryRead(BaseModel):
    id: int
    user_id: Optional[int]
    text: str
    embedding_ref: str
    metadata: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True
