from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..schemas.memory import MemoryCreate, MemoryRead
from ..services.memory_service import memory_service

router = APIRouter(prefix="/memory", tags=["memory"])

@router.post("/", response_model=MemoryRead)
def create_memory(payload: MemoryCreate, db: Session = Depends(get_db)):
    mem = memory_service.create_memory(db, payload)
    return mem

@router.get("/search")
def search_memory(q: str, k: int = 5):
    return {"results": memory_service.search_related(q, k=k)}
