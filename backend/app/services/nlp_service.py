from typing import List
from ..core.config import get_settings
from .memory_service import memory_service
from sqlalchemy.orm import Session
import os

settings = get_settings()

try:  # OpenAI Python SDK v1 style client
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

class NLPService:
    def __init__(self):
        key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        if OpenAI and key:
            try:
                self.client = OpenAI(api_key=key)  # type: ignore
            except Exception:
                self.client = None
        self.available = self.client is not None

    def _build_context(self, db: Session, messages: List[str], include_memories: bool, top_k: int) -> tuple[str, List[str]]:
        used_ids: List[str] = []
        memory_texts: List[str] = []
        if include_memories and messages:
            last_user = messages[-1]
            similar = memory_service.search_related(last_user, k=top_k)
            embedding_ids = [sid for sid, _dist in similar]
            db_mems = memory_service.fetch_memories_by_embedding_ids(db, embedding_ids)
            # preserve order by embedding_ids
            mem_map = {m.embedding_ref: m for m in db_mems}
            for eid in embedding_ids:
                if eid in mem_map:
                    used_ids.append(eid)
                    memory_texts.append(mem_map[eid].text)  # type: ignore[attr-defined]
        context_block = "\n".join(f"[MEMORY] {t}" for t in memory_texts)
        conversation_block = "\n".join(messages[-10:])
        prompt = f"System Persona: {settings.system_persona}\n" + (f"Relevant Memories:\n{context_block}\n\n" if context_block else "") + f"Conversation:\n{conversation_block}\nAssistant Response:"  # noqa: E501
        return prompt, used_ids

    def chat(self, db: Session, messages: List[str], include_memories: bool = True) -> tuple[str, int]:
        # Build contextual prompt
        prompt, used_ids = self._build_context(db, messages, include_memories, settings.memory_top_k)
        if not self.available or not self.client:
            return (f"[offline-model] {prompt[-180:]}" , len(used_ids))

        try:
            chat_messages = [
                {"role": "system", "content": settings.system_persona},
                *[{"role": "user", "content": m} for m in messages[-10:]]
            ]
            completion = self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=settings.openai_model,
                messages=chat_messages,  # type: ignore[arg-type]
                temperature=0.7,
                max_tokens=400,
            )
            content = completion.choices[0].message.content if completion.choices else "(no response)"  # type: ignore
            return (content or "(empty)" ).strip(), len(used_ids)
        except Exception as e:
            return f"[openai-error] {e}. Fallback: {prompt[-160:]}", len(used_ids)

nlp_service = NLPService()
