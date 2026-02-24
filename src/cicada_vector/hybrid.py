"""
Hybrid Database: Merges vector and keyword search results via Reciprocal Rank Fusion.
"""

import os
from typing import List, Tuple, Dict, Optional
from .db import EmbeddingDB
from .keyword_db import KeywordDB
from .embeddings import EmbeddingProvider, OllamaEmbedding


class Store:
    def __init__(
        self,
        storage_dir: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "nomic-embed-text"
    ):
        """
        Initialize hybrid store.

        Args:
            storage_dir: Directory for storing index files
            embedding_provider: Custom embedding provider (if None, uses Ollama)
            ollama_host: Ollama host (used if embedding_provider is None)
            ollama_model: Ollama model (used if embedding_provider is None)
        """
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)

        self.vector_db = EmbeddingDB(os.path.join(storage_dir, "vectors.jsonl"))
        self.keyword_db = KeywordDB(os.path.join(storage_dir, "keywords.json"))

        if embedding_provider is None:
            self.embedder = OllamaEmbedding(host=ollama_host, model=ollama_model)
        else:
            self.embedder = embedding_provider

    def add(self, id: str, text: str, meta: Optional[dict] = None, vector: Optional[List[float]] = None, embed_text: Optional[str] = None):
        """
        Add document to both indexes.

        Args:
            id: Document ID
            text: Text content to index (used for keyword search)
            meta: Optional metadata
            vector: Optional pre-computed vector (if None, will be computed from embed_text or text)
            embed_text: Optional text to use for embedding (if different from text, e.g., truncated)
        """
        if vector is None:
            text_to_embed = embed_text if embed_text is not None else text
            vector = self.embedder.embed(text_to_embed)

        self.vector_db.add(id, vector, meta)
        # IMPORTANT: Always index FULL text for keywords (not truncated)
        self.keyword_db.add(id, text)

    def persist(self):
        """Save both indexes to disk."""
        self.vector_db.persist()
        self.keyword_db.persist()

    def _rrf_merge(
        self,
        vector_results: List[Tuple[str, float, dict]],
        keyword_results: List[str],
        k_const: int = 60
    ) -> List[Tuple[str, float, dict]]:
        """
        Reciprocal Rank Fusion: combines dense and sparse results by rank position.

        Scale-invariant: only cares about rank, not raw score magnitude.
        Both sources contribute equally regardless of their scoring distributions.
        Output normalized to [0, 1] so display scores remain intuitive.
        """
        rrf_scores: Dict[str, float] = {}
        meta_lookup: Dict[str, dict] = {}

        # Dense results: ranked by cosine similarity (descending)
        for rank, (doc_id, _, meta) in enumerate(vector_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 1 + k_const)
            meta_lookup[doc_id] = meta

        # Sparse results: ranked by IDF score from keyword_db (descending)
        for rank, doc_id in enumerate(keyword_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 1 + k_const)
            if doc_id not in meta_lookup:
                try:
                    idx = self.vector_db.ids.index(doc_id)
                    meta_lookup[doc_id] = self.vector_db.metadata[idx]
                except ValueError:
                    meta_lookup[doc_id] = {}

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Normalize to [0, 1] so the top result always reads as 1.0
        if sorted_ids:
            max_score = rrf_scores[sorted_ids[0]]
            if max_score > 0:
                for doc_id in rrf_scores:
                    rrf_scores[doc_id] /= max_score

        return [(doc_id, rrf_scores[doc_id], meta_lookup.get(doc_id, {})) for doc_id in sorted_ids]

    def search(self, query_text: str, k: int = 5, query_vector: Optional[List[float]] = None) -> List[Tuple[str, float, dict]]:
        """
        Perform hybrid search.

        Args:
            query_text: Raw text query (for keywords and embedding)
            k: Number of results
            query_vector: Optional pre-computed query embedding
        """
        if query_vector is None:
            query_vector = self.embedder.embed(query_text)

        # Fetch more than k from each source before fusion
        vector_hits = self.vector_db.search(query_vector, k=k * 3)
        keyword_hits = self.keyword_db.search(query_text)

        merged = self._rrf_merge(vector_hits, keyword_hits)
        return merged[:k]


# Backwards compatibility alias
HybridDB = Store
