"""
Hybrid Database: Merges Vector and Keyword search results.
"""

import os
from typing import List, Tuple, Dict, Any, Optional
from .db import VectorDB
from .keyword_db import KeywordDB

class HybridDB:
    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
        self.vector_db = VectorDB(os.path.join(storage_dir, "vectors.jsonl"))
        self.keyword_db = KeywordDB(os.path.join(storage_dir, "keywords.json"))

    def add(self, id: str, vector: List[float], text: str, meta: Optional[dict] = None):
        """Add document to both indexes."""
        self.vector_db.add(id, vector, meta)
        self.keyword_db.add(id, text)

    def persist(self):
        """Save both indexes to disk."""
        self.vector_db.persist()
        self.keyword_db.persist()

    def _rrf_merge(self, vector_results: List[Tuple[str, float]], keyword_results: List[str], k: int = 60) -> List[Tuple[str, float, dict]]:
        """
        Reciprocal Rank Fusion (RRF)
        Score = 1 / (k + rank_i)
        """
        scores: Dict[str, float] = {}
        
        # 1. Process Vector Results
        # vector_results is list of (id, cosine_score, meta)
        for rank, (doc_id, _, _) in enumerate(vector_results):
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))
            
        # 2. Process Keyword Results
        # keyword_results is just list of ids (unordered/binary match)
        # We treat them all as "Rank 1" matches because exact match is strong
        for doc_id in keyword_results:
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + 1))
            # Boost exact keyword matches further? 
            # RRF is usually robust enough, but let's give keywords a 2x weight
            # effectively treating them as "Rank 0" or very high confidence
            scores[doc_id] += (1 / (k + 1)) 

        # 3. Sort and Attach Metadata
        # We need to look up metadata. It's stored in VectorDB (which is our "primary" store)
        # VectorDB's 'search' returns meta, but for keyword-only matches we might not have it handy easily
        # Optimization: In a real DB, we'd have a separate DocStore. 
        # Here, we assume VectorDB holds the source of truth for metadata.
        
        # Build lookup map from vector results (since we usually have them)
        # If a doc is ONLY found by keywords, we might miss metadata here if we don't scan.
        # For this MVP, we'll scan the VectorDB in-memory list if needed (it's fast).
        
        # Create fast lookup for metadata
        meta_lookup = {}
        for i, doc_id in enumerate(self.vector_db.ids):
            meta_lookup[doc_id] = self.vector_db.metadata[i]
            
        final_results = []
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        for doc_id in sorted_ids:
            score = scores[doc_id]
            meta = meta_lookup.get(doc_id, {})
            final_results.append((doc_id, score, meta))
            
        return final_results

    def search(self, query_text: str, query_vector: List[float], k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Perform hybrid search.
        
        Args:
            query_text: Raw text query (for keywords)
            query_vector: Embedding of query (for vectors)
            k: Number of results
        """
        # 1. Get Semantic Candidates (fetch more than K to allow reranking)
        vector_hits = self.vector_db.search(query_vector, k=k*3)
        
        # 2. Get Exact Keyword Candidates
        keyword_hits = self.keyword_db.search(query_text)
        
        # 3. Merge
        merged = self._rrf_merge(vector_hits, keyword_hits)
        
        return merged[:k]
