"""
Poor Man's RAG: Broad-to-Specific Search
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from .hybrid import HybridDB

class RagDB:
    def __init__(self, storage_dir: str):
        self.db = HybridDB(storage_dir)

    def add_file(self, file_path: str, content: str, vector: List[float], meta: Optional[dict] = None):
        """
        Index a file.
        vector: Embedding of the file's 'summary' or 'representation'.
        """
        doc_id = file_path
        # We index the full content keywords for the KeywordDB
        # We store the full content in meta for retrieval (Poor man's storage)
        # In a real system, you'd read from disk on demand. Here we cache it for speed.
        metadata = meta or {}
        metadata["content"] = content
        metadata["file_path"] = file_path
        
        self.db.add(doc_id, vector, content, metadata)

    def persist(self):
        self.db.persist()

    def _find_best_window(self, content: str, query_terms: List[str], window_size: int = 3) -> Tuple[int, str]:
        """
        Find the line window with the highest density of query terms.
        Returns (start_line_number, text_snippet).
        """
        lines = content.splitlines()
        if not lines:
            return 0, ""

        query_terms = [t.lower() for t in query_terms]
        max_score = -1
        best_start = 0

        # Sliding window density check
        for i in range(len(lines)):
            window = lines[i : i + window_size]
            text = " ".join(window).lower()
            
            # Score = count of query terms present
            score = sum(1 for term in query_terms if term in text)
            
            if score > max_score:
                max_score = score
                best_start = i
            
            # Optimization: If perfect match (all terms), break early? 
            # No, maybe a later window has them closer together. 
            
        snippet = "\n".join(lines[best_start : best_start + window_size])
        return best_start + 1, snippet # 1-based indexing

    def search(self, query: str, query_vector: List[float], k: int = 3) -> List[Dict]:
        """
        1. Find relevant files (Vector + Keyword Search).
        2. Scan files for best line window.
        """
        # 1. Broad Search
        file_results = self.db.search(query, query_vector, k=k)
        
        rag_results = []
        query_terms = re.findall(r"\w+", query)
        
        for doc_id, score, meta in file_results:
            content = meta.get("content", "")
            if not content:
                continue
                
            # 2. Specific Scan
            line_num, snippet = self._find_best_window(content, query_terms)
            
            rag_results.append({
                "file": meta.get("file_path", doc_id),
                "score": score,
                "line": line_num,
                "snippet": snippet,
                "full_match": score > 0.5 # Arbitrary confidence flag
            })
            
        return rag_results
