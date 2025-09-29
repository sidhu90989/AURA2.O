import os
import uuid
from typing import List, Tuple, Optional

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # graceful fallback

import numpy as np
from sentence_transformers import SentenceTransformer
from ..core.config import get_settings

settings = get_settings()

class VectorStore:
    def __init__(self, directory: str | None = None, model_name: str | None = None):
        self.dir = directory or settings.vector_store_dir
        os.makedirs(self.dir, exist_ok=True)
        self.model_name = model_name or settings.embedding_model
        self.model = SentenceTransformer(self.model_name)
        self.index_path = os.path.join(self.dir, "index.faiss")
        self.meta_path = os.path.join(self.dir, "meta.npy")
        self.ids: list[str] = []
        self._load()

    def _load(self):
        if faiss and os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.index = faiss.read_index(self.index_path)
            self.ids = np.load(self.meta_path, allow_pickle=True).tolist()
        else:
            # create new index
            if faiss:
                dummy_vec = self.model.encode(["hello world"])  # shape (1, d)
                d = dummy_vec.shape[1]
                self.index = faiss.IndexFlatL2(d)
            else:
                self.index = None

    def add_texts(self, texts: List[str]) -> List[str]:
        embeddings = self.model.encode(texts)
        new_ids = [str(uuid.uuid4()) for _ in texts]
        if faiss and self.index:
            self.index.add(np.array(embeddings, dtype="float32"))
        # append metadata
        self.ids.extend(new_ids)
        self._persist()
        return new_ids

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        emb = self.model.encode([query])
        if faiss and self.index:
            D, I = self.index.search(np.array(emb, dtype="float32"), k)
            results: List[Tuple[str, float]] = []
            for dist, idx in zip(D[0], I[0]):
                if idx < len(self.ids):
                    results.append((self.ids[idx], float(dist)))
            return results
        # Fallback brute force using numpy cosine
        all_embs = self.model.encode(self.ids) if self.ids else []
        sims: List[Tuple[str, float]] = []
        if len(all_embs):
            q = emb[0]
            for i, (doc_id, vec) in enumerate(zip(self.ids, all_embs)):
                sim = float(np.dot(q, vec) / (np.linalg.norm(q) * np.linalg.norm(vec)))
                sims.append((doc_id, 1 - sim))  # mimic distance
            sims.sort(key=lambda x: x[1])
        return sims[:k]

    def _persist(self):
        if faiss and self.index:
            faiss.write_index(self.index, self.index_path)
        np.save(self.meta_path, np.array(self.ids, dtype=object), allow_pickle=True)
