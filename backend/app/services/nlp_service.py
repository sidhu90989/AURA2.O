from typing import List, Optional
from ..core.config import get_settings
from .memory_service import memory_service
from sqlalchemy.orm import Session
import os

settings = get_settings()

try:
    import openai
except Exception:
    openai = None  # type: ignore

class NLPService:
    def __init__(self):
        self.available = bool(openai and (settings.openai_api_key or os.getenv('OPENAI_API_KEY')))
        if self.available:
            openai.api_key = settings.openai_api_key or os.getenv('OPENAI_API_KEY')

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
                    memory_texts.append(mem_map[eid].text)
        context_block = "\n".join(f"[MEMORY] {t}" for t in memory_texts)
        conversation_block = "\n".join(messages[-10:])
        prompt = f"System Persona: {settings.system_persona}\n" + (f"Relevant Memories:\n{context_block}\n\n" if context_block else "") + f"Conversation:\n{conversation_block}\nAssistant Response:"  # noqa: E501
        return prompt, used_ids

    def chat(self, db: Session, messages: List[str], include_memories: bool = True) -> tuple[str, int]:
        # Build contextual prompt
        prompt, used_ids = self._build_context(db, messages, include_memories, settings.memory_top_k)
        if not self.available:
            return (f"[offline-model] {prompt[-180:]}" , len(used_ids))

        try:
            # Use new Responses or Chat Completions depending on library version
            if hasattr(openai, 'ChatCompletion'):
                completion = openai.ChatCompletion.create(
                    model=settings.openai_model,
                    messages=[
                        {"role": "system", "content": settings.system_persona},
                        *[{"role": "user", "content": m} for m in messages[-10:]]
                    ],
                    temperature=0.7,
                    max_tokens=400
                )
                content = completion['choices'][0]['message']['content']  # type: ignore
            else:
                # Fallback simple completion API
                resp = openai.Completion.create(model=settings.openai_model, prompt=prompt, max_tokens=300)  # type: ignore
                content = resp['choices'][0]['text']  # type: ignore
            return content.strip(), len(used_ids)
        except Exception as e:
            return f"[openai-error] {e}. Fallback: {prompt[-160:]}", len(used_ids)

nlp_service = NLPService()
