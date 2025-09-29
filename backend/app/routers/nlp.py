from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..schemas.chat import ChatRequest, ChatResponse
from ..services.nlp_service import nlp_service
from ..core.database import get_db
from ..services.memory_service import memory_service
from ..schemas.memory import MemoryCreate

router = APIRouter(prefix="/nlp", tags=["nlp"])

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    response, used = nlp_service.chat(db, req.messages, include_memories=req.include_memories)
    # optionally store last user + assistant messages as memories
    if req.store and req.messages:
        try:
            last_user = req.messages[-1]
            memory_service.create_memory(db, MemoryCreate(user_id=req.user_id, text=last_user))
            memory_service.create_memory(db, MemoryCreate(user_id=req.user_id, text=response))
        except Exception:
            pass
    return ChatResponse(response=response, used_memories=used)
