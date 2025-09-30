from typing import Dict, Any, List

try:
    from memory_manager import KnowledgeGraph  # type: ignore  # existing module
except Exception:  # pragma: no cover
    KnowledgeGraph = None  # type: ignore

class KnowledgeGraphService:
    def __init__(self):
        self.available = KnowledgeGraph is not None
        self.graph = None
        if self.available and KnowledgeGraph:  # type: ignore[truthy-function]
            try:
                self.graph = KnowledgeGraph()  # type: ignore[operator]
            except Exception:  # pragma: no cover
                self.available = False
                self.graph = None

    def store(self, data: Dict[str, Any]) -> str:
        if not (self.available and self.graph):
            return "GRAPH_UNAVAILABLE"
        return self.graph.store_context(data) or "ERROR"

    def related(self, text: str, limit: int = 5) -> List[Any]:
        if not (self.available and self.graph):
            return []
        return self.graph.get_related_contexts(text, limit=limit)

    def by_emotion(self, emotion: str):
        if not (self.available and self.graph):
            return []
        return self.graph.query_by_emotion(emotion)

knowledge_graph_service = KnowledgeGraphService()
