"""Vector-based memory using sentence embeddings and FAISS."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

import faiss
from sentence_transformers import SentenceTransformer


class VectorMemory:
    """Store textual memories and retrieve them by semantic similarity."""

    def __init__(self, path: str = "vec_memory") -> None:
        self.index_path = Path(path).with_suffix(".index")
        self.text_path = Path(path).with_suffix(".txt")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        dim = self.model.get_sentence_embedding_dimension()
        self.embeddings: list[np.ndarray] = []
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.texts = self.text_path.read_text().splitlines()
            if self.texts:
                self.embeddings = self.model.encode(self.texts, convert_to_numpy=True).tolist()
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.texts = []

    def add(self, text: str) -> None:
        emb = self.model.encode([text], convert_to_numpy=True)
        self.index.add(emb)
        self.texts.append(text)
        self.embeddings.append(emb[0])
        faiss.write_index(self.index, str(self.index_path))
        with open(self.text_path, "a", encoding="utf-8") as f:
            f.write(text.replace("\n", " ") + "\n")

    def search(self, query: str, top_k: int = 3, lambda_param: float = 0.5) -> List[str]:
        """Return diverse results using Maximal Marginal Relevance."""
        if not self.texts:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)[0]
        n_candidates = min(top_k * 5, len(self.texts))
        scores, indices = self.index.search(q_emb.reshape(1, -1), n_candidates)
        cand_indices = [i for i in indices[0] if i != -1]
        selected: list[int] = []
        while len(selected) < top_k and cand_indices:
            best_idx = None
            best_score = -1e9
            for idx in cand_indices:
                if idx in selected:
                    continue
                cand = self.embeddings[idx]
                relevance = float(np.dot(q_emb, cand) / (np.linalg.norm(q_emb) * np.linalg.norm(cand)))
                diversity = 0.0
                if selected:
                    div_sims = [float(np.dot(cand, self.embeddings[j]) / (np.linalg.norm(cand) * np.linalg.norm(self.embeddings[j]))) for j in selected]
                    diversity = max(div_sims)
                score = lambda_param * relevance - (1 - lambda_param) * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            if best_idx is None:
                break
            selected.append(best_idx)
            cand_indices.remove(best_idx)
        return [self.texts[i] for i in selected]
