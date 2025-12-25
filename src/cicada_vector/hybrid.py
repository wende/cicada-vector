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

    def _score_boosting_merge(self, vector_results: List[Tuple[str, float]], keyword_results: List[str]) -> List[Tuple[str, float, dict]]:
        """
        Merge results using Score Boosting.
        Base score = Cosine Similarity.
        Keyword match = Boost (+0.2).
        """
        # Map: doc_id -> score
        final_scores: Dict[str, float] = {}
        
        # 1. Start with Vector Scores (True Semantic Confidence)
        # vector_results is list of (id, cosine_score, meta)
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
                
        # 3. Add Keyword-Only Matches
        # If found by keyword but NOT by vector (in top K), give it a base score
        # A pure keyword match is usually strong, let's say 0.5 minimum confidence
        for doc_id in keyword_results:
            if doc_id not in final_scores:
                final_scores[doc_id] = 0.5
                # We need to fetch metadata for these since we didn't get it from vector search
                # In this MVP we rely on vector_db having everything loaded
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
        
        # 3. Merge using Score Boosting
        merged = self._score_boosting_merge(vector_hits, keyword_hits)
        
        return merged[:k]
