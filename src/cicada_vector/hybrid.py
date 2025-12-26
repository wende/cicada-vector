"""
Hybrid Database: Merges Vector and Keyword search results.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
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

        # Set up embedding provider
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
        # Get embedding if not provided
        if vector is None:
            # Use embed_text for embedding if provided, otherwise use full text
            text_to_embed = embed_text if embed_text is not None else text
            vector = self.embedder.embed(text_to_embed)

        self.vector_db.add(id, vector, meta)
        # IMPORTANT: Always index FULL text for keywords (not truncated)
        self.keyword_db.add(id, text)

    def persist(self):
        """Save both indexes to disk."""
        self.vector_db.persist()
        self.keyword_db.persist()

    def _score_boosting_merge(self, vector_results: List[Tuple[str, float, dict]], keyword_results: List[str]) -> List[Tuple[str, float, dict]]:
        """
        Merge results using Score Boosting.
        Base score = Cosine Similarity.
        Keyword match = Boost (+0.2).
        """
        # Map: doc_id -> score
        final_scores: Dict[str, float] = {}

        # 1. Start with Vector Scores (True Semantic Confidence)
        meta_lookup = {}

        for doc_id, score, meta in vector_results:
            final_scores[doc_id] = score
            meta_lookup[doc_id] = meta

        # 2. Apply Keyword Boosts
        keyword_set = set(keyword_results)

        # Boost vector matches that also have keyword matches
        for doc_id in final_scores:
            if doc_id in keyword_set:
                # Boost by 20% (clamped to 1.0)
                final_scores[doc_id] = min(1.0, final_scores[doc_id] + 0.2)

        # 3. Add Keyword-Only Matches (lower confidence range)
        for doc_id in keyword_results:
            if doc_id not in final_scores:
                # Keyword-only matches get 0.4-0.6 range based on position
                base_score = 0.5 - (keyword_results.index(doc_id) * 0.02)
                final_scores[doc_id] = max(0.4, base_score)
                # Fetch metadata from vector_db
                try:
                    idx = self.vector_db.ids.index(doc_id)
                    meta_lookup[doc_id] = self.vector_db.metadata[idx]
                except ValueError:
                    meta_lookup[doc_id] = {}

        # 4. Sort
        final_results = []
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

        for doc_id in sorted_ids:
            score = final_scores[doc_id]
            meta = meta_lookup.get(doc_id, {})
            final_results.append((doc_id, score, meta))

        return final_results

    def search(self, query_text: str, k: int = 5, query_vector: Optional[List[float]] = None) -> List[Tuple[str, float, dict]]:
        """
        Perform hybrid search.

        Args:
            query_text: Raw text query (for keywords and embedding)
            k: Number of results
            query_vector: Optional pre-computed query embedding
        """
        # Get query embedding if not provided
        if query_vector is None:
            query_vector = self.embedder.embed(query_text)

        # 1. Get Semantic Candidates (fetch more than K to allow reranking)
        vector_hits = self.vector_db.search(query_vector, k=k*3)

        # 2. Get Exact Keyword Candidates
        keyword_hits = self.keyword_db.search(query_text)

        # 3. Merge using Score Boosting
        merged = self._score_boosting_merge(vector_hits, keyword_hits)

        return merged[:k]


# Backwards compatibility alias
HybridDB = Store
