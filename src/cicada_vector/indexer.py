"""
Incremental Directory Indexer for Cicada Vector.
Handles file scanning, hashing, and incremental updates.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Set, Dict, Optional

from .rag import VectorIndex
from .embeddings import EmbeddingProvider

DEFAULT_EXTENSIONS = {'.py', '.ex', '.exs', '.md', '.json', '.toml', '.sh', '.rs', '.go', '.js', '.ts', '.tsx'}
DEFAULT_EXCLUDE = {'.git', '.venv', '__pycache__', 'node_modules', '.pytest_cache', '.gemini', 'target', 'dist', 'build'}

class DirectoryIndexer:
    def __init__(
        self,
        storage_dir: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "nomic-embed-text"
    ):
        """
        Initialize directory indexer.

        Args:
            storage_dir: Directory for storage files
            embedding_provider: Custom embedding provider (if None, uses Ollama)
            ollama_host: Ollama host (used if embedding_provider is None)
            ollama_model: Ollama model (used if embedding_provider is None)
        """
        self.storage_dir = Path(storage_dir)
        self.rag_db = VectorIndex(
            str(storage_dir),
            embedding_provider=embedding_provider,
            ollama_host=ollama_host,
            ollama_model=ollama_model
        )

        self.hashes_path = self.storage_dir / "file_hashes.json"
        self.hashes: Dict[str, str] = {}
        self._load_hashes()

    def _load_hashes(self):
        if self.hashes_path.exists():
            try:
                with open(self.hashes_path, 'r') as f:
                    self.hashes = json.load(f)
            except Exception:
                self.hashes = {}

    def _save_hashes(self):
        with open(self.hashes_path, 'w') as f:
            json.dump(self.hashes, f, indent=2)

    def _compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def index_directory(
        self,
        root_path: str,
        extensions: Set[str] = DEFAULT_EXTENSIONS,
        exclude: Set[str] = DEFAULT_EXCLUDE,
        verbose: bool = False
    ) -> Dict[str, int]:
        """
        Incrementally index a directory.
        Returns stats: {'added': int, 'skipped': int, 'failed': int}
        """
        root = Path(root_path).resolve()
        if not root.exists():
            raise FileNotFoundError(f"Path {root} does not exist")

        stats = {'added': 0, 'skipped': 0, 'failed': 0}
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3

        files_to_process = []

        # 1. Scan files
        for p in root.rglob("*"):
            if p.is_file() and p.suffix in extensions:
                # Check exclusions
                if any(part in exclude for part in p.parts):
                    continue
                # Don't index our own DB
                if str(self.storage_dir.name) in p.parts:
                    continue
                files_to_process.append(p)

        if verbose:
            print(f"Scanning {len(files_to_process)} files...", file=sys.stderr)

        # 2. Process files
        for i, file_path in enumerate(files_to_process):
            rel_path = str(file_path.relative_to(root))

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                if not content.strip():
                    continue

                # Check hash
                current_hash = self._compute_hash(content)
                if rel_path in self.hashes and self.hashes[rel_path] == current_hash:
                    stats['skipped'] += 1
                    if verbose:
                        print(f"\r[{i+1}/{len(files_to_process)}] Skipped {rel_path} (Unchanged)", end="", flush=True, file=sys.stderr)
                    continue

                # Embed and Index
                if verbose:
                    print(f"\r[{i+1}/{len(files_to_process)}] Indexing {rel_path}...", end="", flush=True, file=sys.stderr)

                # Add to DB (embedding handled internally by VectorIndex)
                self.rag_db.add_file(rel_path, content)

                # Update hash
                self.hashes[rel_path] = current_hash
                stats['added'] += 1
                consecutive_failures = 0  # Reset on success

            except Exception as e:
                stats['failed'] += 1
                consecutive_failures += 1

                if verbose:
                    print(f"\nError processing {rel_path}: {e}", file=sys.stderr)

                # Stop if we hit too many consecutive failures (likely Ollama issue)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    error_msg = f"\nStopped after {consecutive_failures} consecutive embedding failures.\n"
                    if "500" in str(e) or "Internal Server Error" in str(e):
                        error_msg += "Ollama returned HTTP 500 errors. This usually means:\n"
                        error_msg += "  - File content is too long for embedding\n"
                        error_msg += "  - Ollama model is having issues (try: ollama pull nomic-embed-text)\n"
                    elif "Connection" in str(e) or "refused" in str(e):
                        error_msg += "Cannot connect to Ollama. Is it running? (ollama serve)\n"
                    else:
                        error_msg += f"Error: {e}\n"

                    print(error_msg, file=sys.stderr)
                    break

        # 3. Finalize
        self.rag_db.persist()
        self._save_hashes()

        if verbose:
            print(f"\n\nDone. Added: {stats['added']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}", file=sys.stderr)

        return stats
