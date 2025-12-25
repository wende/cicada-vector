"""
Poor Man's RAG: Broad-to-Specific Search
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from .hybrid import Store
from .embeddings import EmbeddingProvider

class VectorIndex:
    def __init__(
        self,
        storage_dir: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "nomic-embed-text"
    ):
        """
        Initialize RAG index.

        Args:
            storage_dir: Directory for storage files
            embedding_provider: Custom embedding provider (if None, uses Ollama)
            ollama_host: Ollama host (used if embedding_provider is None)
            ollama_model: Ollama model (used if embedding_provider is None)
        """
        self.db = Store(
            storage_dir,
            embedding_provider=embedding_provider,
            ollama_host=ollama_host,
            ollama_model=ollama_model
        )

    def add_file(self, file_path: str, content: str, meta: Optional[dict] = None, vector: Optional[List[float]] = None):
        """
        Index a file.

        Args:
            file_path: Path to the file
            content: File content to index
            meta: Optional metadata
            vector: Optional pre-computed vector (if None, will be computed from content)
        """
        doc_id = file_path
        # We index the full content keywords for the KeywordDB
        # We store the full content in meta for retrieval (Poor man's storage)
        # In a real system, you'd read from disk on demand. Here we cache it for speed.
        metadata = meta or {}
        metadata["content"] = content
        metadata["file_path"] = file_path

        # Truncate content for embedding if needed
        # nomic-embed-text has ~512 token limit (~1000 chars to be safe)
        MAX_CHARS = 900
        embed_content = content
        if len(content) > MAX_CHARS:
            # For embedding, use file path + beginning of content
            # This gives the model context about what the file is
            embed_content = f"File: {file_path}\n\n{content[:MAX_CHARS - len(file_path) - 10]}"

        self.db.add(doc_id, embed_content, metadata, vector=vector)

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

    def search(self, query: str, k: int = 3, query_vector: Optional[List[float]] = None) -> List[Dict]:
        """
        1. Find relevant files (Vector + Keyword Search).
        2. Scan files for best line window.

        Args:
            query: Search query text
            k: Number of results
            query_vector: Optional pre-computed query vector (if None, will be computed from query)

        Returns:
            List of dicts with file, score, line, snippet, full_match
        """
        # 1. Broad Search
        file_results = self.db.search(query, k=k, query_vector=query_vector)
        
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
