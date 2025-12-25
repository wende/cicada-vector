"""
A poor man's vector database.
Zero dependencies. JSONL storage.
"""

import json
import math
import os
from typing import Any, List, Optional, Tuple, Union

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False


class VectorDB:
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
        
        # Load existing data
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
                    self.vectors.append(record['vector'])
                    self.metadata.append(record.get('meta', {}))
                except json.JSONDecodeError:
                    continue
        
        if HAS_NUMPY and self.vectors:
            self._numpy_vectors = np.array(self.vectors, dtype=np.float32)

    def add(self, id: str, vector: List[float], meta: Optional[dict] = None):
        """
        Add a vector to the database.
        
        Args:
            id: Unique identifier for the vector.
            vector: List of floats.
            meta: Optional metadata dict.
        """
        self.ids.append(id)
        self.vectors.append(vector)
        self.metadata.append(meta or {})
        self._dirty = True
        
        # Append to file immediately (Write Ahead Log style)
        record = {
            'id': id,
            'vector': vector,
            'meta': meta or {}
        }
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
            
        # Update numpy cache if it exists (inefficient for single adds, better to batch)
        if HAS_NUMPY and self._numpy_vectors is not None:
            # For simplicity in this poor-man's version, we invalidate the cache
            # It will be rebuilt on next search
            self._numpy_vectors = None

    def persist(self):
        """
        Force save to disk (rewrite entire file).
        Useful if we implement delete/update later.
        For append-only 'add', this isn't strictly necessary as we write on add.
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
        Search for nearest neighbors using Cosine Similarity.
        
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
        """Optimized search using Numpy."""
        if self._numpy_vectors is None:
            self._numpy_vectors = np.array(self.vectors, dtype=np.float32)
            
        q = np.array(query, dtype=np.float32)
        
        # Cosine Similarity: (A . B) / (||A|| * ||B||)
        
        # 1. Norm of query
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
            
        # 2. Norm of all vectors (can be cached, but fast enough for small N)
        v_norms = np.linalg.norm(self._numpy_vectors, axis=1)
        
        # 3. Dot product
        dot_products = np.dot(self._numpy_vectors, q)
        
        # 4. Similarity scores
        # Avoid division by zero
        v_norms[v_norms == 0] = 1e-10
        scores = dot_products / (v_norms * q_norm)
        
        # 5. Top K
        # argpartition is faster than sort for top K
        if k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            
        results = []
        for idx in top_indices:
            results.append((self.ids[idx], float(scores[idx]), self.metadata[idx]))
            
        return results

    def _search_python(self, query: List[float], k: int) -> List[Tuple[str, float, dict]]:
        """Pure Python search implementation."""
        # Precompute query magnitude
        q_mag = math.sqrt(sum(x*x for x in query))
        if q_mag == 0:
            return []
            
        scores = []
        for i, vec in enumerate(self.vectors):
            # Dot product
            dot = sum(q * v for q, v in zip(query, vec))
            
            # Vector magnitude
            v_mag = math.sqrt(sum(x*x for x in vec))
            
            if v_mag == 0:
                score = 0.0
            else:
                score = dot / (q_mag * v_mag)
                
            scores.append((score, i))
            
        # Sort and take top k
        scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scores[:k]:
            results.append((self.ids[idx], score, self.metadata[idx]))
            
        return results
