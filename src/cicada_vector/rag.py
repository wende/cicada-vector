"""
VectorIndex: Two-Level RAG (Broad-to-Specific Search)
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from .db import _normalize_vector
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
        Initialize VectorIndex with two-level RAG architecture.

        Args:
            storage_dir: Directory for storing index files
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

    def prepare_embed_text(self, file_path: str, content: str) -> str:
        """Compute the truncated embed text for a file."""
        max_chars = self.db.embedder.max_chars
        prefix = f"File: {file_path}\n\n"
        available = max_chars - len(prefix)
        if len(content) > available:
            return prefix + content[:available]
        return content

    def add_file(
        self,
        file_path: str,
        content: str,
        meta: Optional[dict] = None,
        vector: Optional[List[float]] = None,
        embed_text: Optional[str] = None
    ):
        """
        Index a file (file-level only). Chunks created on-demand during search.

        Args:
            file_path: Path to the file
            content: File content to index (full text for keywords)
            meta: Optional metadata
            vector: Optional pre-computed vector (if None, will be computed from embed_text or content)
            embed_text: Optional text for embedding (if None, will auto-truncate content)
        """
        # Backwards compatibility:
        # Historical signature was add_file(file_path, content, vector, meta=None, embed_text=None).
        if isinstance(meta, list) and all(isinstance(x, (int, float)) for x in meta):
            legacy_vector = meta
            legacy_meta = vector if isinstance(vector, dict) else None
            meta = legacy_meta
            vector = legacy_vector

        metadata = meta or {}
        metadata["content"] = content
        metadata["file_path"] = file_path

        doc_id = file_path
        if embed_text is None:
            file_embed_text = self.prepare_embed_text(file_path, content)
        else:
            file_embed_text = embed_text

        self.db.add(doc_id, text=content, meta=metadata, vector=vector, embed_text=file_embed_text)

    def persist(self):
        """Save indexes to disk."""
        self.db.persist()

    def _find_all_windows(self, content: str, query_terms: List[str], window_size: int = 3) -> List[Tuple[int, str, int]]:
        """
        Find ALL line windows containing query terms.
        Returns list of (start_line_number, text_snippet, match_score) sorted by score descending.
        """
        lines = content.splitlines()
        if not lines:
            return []

        query_terms = [t.lower() for t in query_terms]
        windows = []

        for i in range(len(lines)):
            window = lines[i : i + window_size]
            text = " ".join(window).lower()

            score = sum(1 for term in query_terms if term in text)

            if score > 0:
                snippet = "\n".join(window)
                windows.append((i + 1, snippet, score))  # 1-based indexing

        windows.sort(key=lambda x: x[2], reverse=True)

        # Deduplicate overlapping windows (keep higher scoring one)
        deduplicated = []
        used_ranges = []

        for line_num, snippet, score in windows:
            line_idx = line_num - 1
            overlaps = any(
                abs(line_idx - used_start) < window_size
                for used_start in used_ranges
            )

            if not overlaps:
                deduplicated.append((line_num, snippet, score))
                used_ranges.append(line_idx)

        return deduplicated

    def _find_best_chunks_for_file(
        self,
        content: str,
        file_path: str,
        query: str,
        query_vector: Optional[List[float]] = None
    ) -> List[Tuple[int, str, float]]:
        """
        Find best matching chunks for a file, creating and caching embeddings on first search.

        Args:
            content: Full file content
            file_path: File path (for context in embeddings)
            query: Search query
            query_vector: Pre-normalized query vector (unit length)

        Returns:
            List of (line_num, chunk_text, score)
        """
        lines = content.splitlines()
        if len(lines) <= 10:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and len(stripped) > 15:
                    return [(i + 1, stripped[:200], 0.0)]
            return []

        WINDOW_SIZE = 15
        STRIDE = 15
        chunks_to_score = []
        chunks_to_create = []

        for i in range(0, len(lines), STRIDE):
            window_lines = lines[i:i + WINDOW_SIZE]
            if len(window_lines) < 3:
                break

            chunk_id = f"{file_path}#chunk_{i}"
            chunk_text = "\n".join(window_lines)

            try:
                idx = self.db.vector_db.ids.index(chunk_id)
                # Stored vectors are pre-normalized
                chunk_vector = self.db.vector_db.vectors[idx]
                chunk_meta = self.db.vector_db.metadata[idx]
                chunks_to_score.append((i + 1, chunk_text, chunk_vector, chunk_id, chunk_meta))
            except ValueError:
                chunks_to_create.append((i, chunk_text, chunk_id))

        if query_vector is None:
            raw = self.db.embedder.embed(query)
            query_vector = _normalize_vector(raw)

        if chunks_to_create:
            for i, chunk_text, chunk_id in chunks_to_create:
                chunk_embed_text = f"File: {file_path} (lines {i+1}-{min(i+WINDOW_SIZE, len(lines))})\n\n{chunk_text}"
                raw_vector = self.db.embedder.embed(chunk_embed_text)
                # Normalize before storing and scoring
                chunk_vector = _normalize_vector(raw_vector)
                chunk_meta = {
                    "file_path": file_path,
                    "start_line": i + 1,
                    "end_line": min(i + WINDOW_SIZE, len(lines)),
                    "chunk_text": chunk_text,
                    "is_chunk": True
                }
                self.db.vector_db.add(chunk_id, chunk_vector, chunk_meta)
                chunks_to_score.append((i + 1, chunk_text, chunk_vector, chunk_id, chunk_meta))
            self.db.vector_db.persist()

        # Score chunks: both query and stored vectors are unit-length, so dot product = cosine similarity
        scored_chunks = []
        for line_num, chunk_text, chunk_vector, chunk_id, chunk_meta in chunks_to_score:
            similarity = sum(q * v for q, v in zip(query_vector, chunk_vector))
            scored_chunks.append((line_num, chunk_text, similarity))

        scored_chunks.sort(key=lambda x: x[2], reverse=True)
        return scored_chunks[:1]

    def search(self, query: str, query_vector: Optional[List[float]] = None, k: int = 3) -> List[Dict]:
        """
        Two-level search:
        1. Broad Search: Find relevant files (file-level embeddings + keywords)
        2. Specific Search: Find best chunks/windows within matched files

        Args:
            query: Search query
            k: Number of files to return
            query_vector: Optional pre-computed query vector

        Returns:
            List of dicts with file, score, line, snippet, full_match
        """
        # Backwards compatibility:
        # Historical signature was search(query, query_vector, k=3).
        if isinstance(k, list) and all(isinstance(x, (int, float)) for x in k):
            query_vector = k
            k = 3

        if query_vector is None:
            query_vector = self.db.embedder.embed(query)
        # Normalize once; reused for both Store.search and chunk similarity
        query_vector = _normalize_vector(query_vector)

        # Fetch more results to account for cached chunks in the index
        all_results = self.db.search(query, k=k * 10, query_vector=query_vector)

        rag_results = []
        query_terms = re.findall(r"\w+", query)
        seen_files = set()
        files_processed = 0

        for doc_id, file_score, meta in all_results:
            # Skip cached chunks — only want file-level docs in broad search
            if '#chunk_' in doc_id:
                continue

            file_path = meta.get("file_path", doc_id)
            if file_path in seen_files:
                continue
            seen_files.add(file_path)

            content = meta.get("content", "")
            if not content:
                continue

            keyword_windows = self._find_all_windows(content, query_terms)

            if keyword_windows:
                for line_num, snippet, window_score in keyword_windows:
                    rag_results.append({
                        "file": file_path,
                        "score": file_score,
                        "line": line_num,
                        "snippet": snippet,
                        "full_match": True
                    })
            else:
                best_chunks = self._find_best_chunks_for_file(content, file_path, query, query_vector)

                for line_num, snippet, chunk_score in best_chunks:
                    combined_score = (file_score + chunk_score) / 2
                    rag_results.append({
                        "file": file_path,
                        "score": combined_score,
                        "line": line_num,
                        "snippet": snippet,
                        "full_match": False
                    })

            files_processed += 1
            if files_processed >= k:
                break

        rag_results.sort(key=lambda x: x["score"], reverse=True)
        return rag_results


# Backwards compatibility
RagDB = VectorIndex
