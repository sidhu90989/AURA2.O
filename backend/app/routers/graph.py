from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from ..services.knowledge_graph_service import knowledge_graph_service

router = APIRouter(prefix="/graph", tags=["graph"])

class GraphStoreRequest(BaseModel):
    text: str
    emotion: str = Field(default="neutral")
    user: Optional[str] = None
    intent: Optional[str] = None

@router.post('/store')
async def store(req: GraphStoreRequest):
    return {"id": knowledge_graph_service.store(req.model_dump())}

@router.get('/related')
async def related(q: str, limit: int = 5):
    return {"results": knowledge_graph_service.related(q, limit=limit)}

@router.get('/emotion')
async def by_emotion(emotion: str):
    return {"results": knowledge_graph_service.by_emotion(emotion)}
