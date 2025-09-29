from sqlalchemy.orm import Session
from ..models.memory import Memory
from ..schemas.memory import MemoryCreate
from .vectorstore import VectorStore
from typing import List, Tuple

vector_store = VectorStore()

class MemoryService:
    def __init__(self):
        self.store = vector_store

    def create_memory(self, db: Session, data: MemoryCreate) -> Memory:
        ids = self.store.add_texts([data.text])
        mem = Memory(
            user_id=data.user_id,
            text=data.text,
            embedding_ref=ids[0],
            metadata=data.metadata or {}
        )
        db.add(mem)
        db.commit()
        db.refresh(mem)
        return mem

    def search_related(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        return self.store.similarity_search(query, k=k)

    def fetch_memories_by_embedding_ids(self, db: Session, embedding_ids: List[str]) -> List[Memory]:
        if not embedding_ids:
            return []
        return db.query(Memory).filter(Memory.embedding_ref.in_(embedding_ids)).all()

memory_service = MemoryService()
