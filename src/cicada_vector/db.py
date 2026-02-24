"""
A poor man's vector database.
Zero dependencies. JSONL storage.
"""

import json
import math
import os
from typing import List, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


def _normalize_vector(vector: List[float]) -> List[float]:
    """Return unit-length version of vector."""
    if HAS_NUMPY:
        v = np.array(vector, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm == 0:
            return list(vector)
        return (v / norm).tolist()
    else:
        mag = math.sqrt(sum(x * x for x in vector))
        if mag == 0:
            return list(vector)
        return [x / mag for x in vector]


class EmbeddingDB:
    def __init__(self, storage_path: str):
        """
        Initialize the vector database.

        Args:
            storage_path: Path to the .jsonl file for storage.
        """
        self.storage_path = storage_path
        self.vectors: List[List[float]] = []
        self.ids: List[str] = []
        self.metadata: List[dict] = []
        self._numpy_vectors = None
        self._dirty = False

        if os.path.exists(storage_path):
            self._load()

    def _load(self):
        """Load data from JSONL file."""
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self.ids.append(record['id'])
                    # Normalize on load for backward compat with un-normalized indices
                    self.vectors.append(_normalize_vector(record['vector']))
                    raw_meta = record.get('meta', {})
                    self.metadata.append(raw_meta if isinstance(raw_meta, dict) else {})
                except json.JSONDecodeError:
                    continue

        if HAS_NUMPY and self.vectors:
            self._numpy_vectors = np.array(self.vectors, dtype=np.float32)

    def add(self, id: str, vector: List[float], meta: Optional[dict] = None):
        """
        Add a vector to the database. Vector is normalized to unit length.

        Args:
            id: Unique identifier for the vector.
            vector: List of floats (normalized to unit length before storage).
            meta: Optional metadata dict.
        """
        normalized = _normalize_vector(vector)
        self.ids.append(id)
        self.vectors.append(normalized)
        meta_dict = meta if isinstance(meta, dict) else {}
        self.metadata.append(meta_dict)
        self._dirty = True

        # Append to file immediately (Write Ahead Log style)
        record = {
            'id': id,
            'vector': normalized,
            'meta': meta_dict
        }
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

        if HAS_NUMPY and self._numpy_vectors is not None:
            self._numpy_vectors = None

    def persist(self):
        """
        Force save to disk (rewrite entire file).
        """
        if not self._dirty:
            return

        with open(self.storage_path, 'w', encoding='utf-8') as f:
            for i in range(len(self.ids)):
                record = {
                    'id': self.ids[i],
                    'vector': self.vectors[i],
                    'meta': self.metadata[i]
                }
                f.write(json.dumps(record) + '\n')
        self._dirty = False

    def search(self, query: List[float], k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search for nearest neighbors using cosine similarity.
        Vectors are pre-normalized, so this is a dot product search.

        Args:
            query: Query vector.
            k: Number of results to return.

        Returns:
            List of (id, score, metadata) sorted by score descending.
        """
        if not self.vectors:
            return []

        if HAS_NUMPY:
            return self._search_numpy(query, k)
        else:
            return self._search_python(query, k)

    def _search_numpy(self, query: List[float], k: int) -> List[Tuple[str, float, dict]]:
        """Optimized search using NumPy. Pre-normalized vectors make this a pure dot product."""
        if self._numpy_vectors is None:
            self._numpy_vectors = np.array(self.vectors, dtype=np.float32)

        q = np.array(query, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm  # normalize query

        # Dot product on pre-normalized stored vectors = cosine similarity
        scores = np.dot(self._numpy_vectors, q)

        if k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self.ids[idx], float(scores[idx]), self.metadata[idx]) for idx in top_indices]

    def _search_python(self, query: List[float], k: int) -> List[Tuple[str, float, dict]]:
        """Pure Python search. Pre-normalized vectors make this a pure dot product."""
        mag = math.sqrt(sum(x * x for x in query))
        if mag == 0:
            return []
        q_norm = [x / mag for x in query]  # normalize query

        scores = []
        for i, vec in enumerate(self.vectors):
            # Dot product on pre-normalized stored vectors = cosine similarity
            score = sum(q * v for q, v in zip(q_norm, vec))
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        return [(self.ids[idx], score, self.metadata[idx]) for score, idx in scores[:k]]


# Backwards compatibility alias
VectorDB = EmbeddingDB
