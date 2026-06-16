# recognition/faiss_db.py
"""
FAISS IndexFlatIP (inner product = cosine similarity for L2-normalised vectors).
Supports incremental addition of new identities without rebuilding.
"""

import faiss
import numpy as np
import pickle
from pathlib import Path


class FaceDatabase:
    """
    FAISS IndexFlatIP (inner product = cosine similarity for L2-normalised vectors).
    Supports incremental addition of new identities without rebuilding.
    """

    def __init__(self, dim=512, index_path=None, meta_path=None):
        self.dim = dim
        self.index_path = Path(index_path) if index_path else None
        self.meta_path = Path(meta_path) if meta_path else None
        self.index = faiss.IndexFlatIP(dim)          # Inner Product
        self.labels: list[str] = []                  # Parallel list to FAISS vectors
        self._load_if_exists()

    def _load_if_exists(self):
        if self.index_path and self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            print(f"[FAISS] Loaded index with {self.index.ntotal} vectors")
        if self.meta_path and self.meta_path.exists():
            with open(self.meta_path, 'rb') as f:
                self.labels = pickle.load(f)
            print(f"[FAISS] Loaded {len(self.labels)} labels for {len(set(self.labels))} persons")

    def save(self):
        if self.index_path:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
        if self.meta_path:
            self.meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.meta_path, 'wb') as f:
                pickle.dump(self.labels, f)

    def add_person(self, name: str, embeddings: np.ndarray):
        """
        Add multiple embeddings for one person.
        Each embedding is stored as a separate FAISS entry with the same label.
        Majority vote during search handles per-embedding results.
        """
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.dim, \
            f"Expected shape (N, {self.dim}), got {embeddings.shape}"
        self.index.add(embeddings)
        self.labels.extend([name] * len(embeddings))
        self.save()

    def search(self, query: np.ndarray, top_k=5) -> list[tuple[str, float]]:
        """
        Returns list of (name, cosine_score) for top_k candidates.
        query: 1-D normalised vector of shape (dim,)
        """
        if self.index.ntotal == 0:
            return []
        q = query.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.labels[idx], float(score)))
        return results

    def majority_vote_search(self, query: np.ndarray, top_k=7) -> tuple[str, float]:
        """
        Search top_k, then pick the name with highest cumulative score.
        More robust than single nearest-neighbour for multiple enrolled images.
        """
        effective_k = min(top_k, max(1, self.index.ntotal // 2))
        candidates = self.search(query, top_k=effective_k)
        if not candidates:
            return "Unknown", 0.0

        vote_scores: dict[str, float] = {}
        vote_counts: dict[str, int] = {}
        for name, score in candidates:
            vote_scores[name] = vote_scores.get(name, 0.0) + score
            vote_counts[name] = vote_counts.get(name, 0) + 1

        best_name = max(vote_scores, key=vote_scores.__getitem__)
        best_score = vote_scores[best_name] / vote_counts[best_name]
        return best_name, best_score

    @property
    def enrolled_names(self) -> set:
        return set(self.labels)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal
