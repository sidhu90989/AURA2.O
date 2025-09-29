from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    messages: List[str] = Field(..., min_length=1)
    user_id: Optional[int] = None
    store: bool = True
    include_memories: bool = True

class ChatResponse(BaseModel):
    response: str
    used_memories: int | None = None
