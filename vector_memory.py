"""Vector-based memory using sentence embeddings and FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer


class VectorMemory:
    """Store textual memories and retrieve them by semantic similarity."""

    def __init__(self, path: str = "vec_memory") -> None:
        self.index_path = Path(path).with_suffix(".index")
        self.text_path = Path(path).with_suffix(".txt")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        dim = self.model.get_sentence_embedding_dimension()
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.texts = self.text_path.read_text().splitlines()
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.texts = []

    def add(self, text: str) -> None:
        emb = self.model.encode([text], convert_to_numpy=True)
        self.index.add(emb)
        self.texts.append(text)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(text.replace("\n", " ") + "\n")

    def search(self, query: str, top_k: int = 3) -> List[str]:
        if not self.texts:
            return []
        emb = self.model.encode([query], convert_to_numpy=True)
        scores, indices = self.index.search(emb, top_k)
        return [self.texts[i] for i in indices[0] if i != -1]
