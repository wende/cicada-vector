"""
Git Commit Indexer for Cicada Vector.
Indexes git commits for semantic search.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List

from .rag import VectorIndex
from .embeddings import EmbeddingProvider


class GitIndexer:
    def __init__(
        self,
        storage_dir: str,
        embedding_provider: Optional[EmbeddingProvider] = None,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "nomic-embed-text"
    ):
        """
        Initialize git commit indexer.

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

        self.hashes_path = self.storage_dir / "commit_hashes.json"
        self.indexed_commits: Dict[str, str] = {}
        self._load_indexed()

    def _load_indexed(self):
        """Load already indexed commits."""
        if self.hashes_path.exists():
            try:
                with open(self.hashes_path, 'r') as f:
                    self.indexed_commits = json.load(f)
            except Exception:
                self.indexed_commits = {}

    def _save_indexed(self):
        """Save indexed commit hashes."""
        with open(self.hashes_path, 'w') as f:
            json.dump(self.indexed_commits, f, indent=2)

    def _get_commits(self, repo_path: Path, limit: int = 100, since: Optional[str] = None) -> List[Dict]:
        """
        Get commits from git log.

        Args:
            repo_path: Path to git repository
            limit: Maximum number of commits to fetch
            since: Date string (e.g., '2024-01-01', '7 days ago')

        Returns:
            List of commit dictionaries
        """
        # Format: sha|author|email|date|subject
        # Then followed by body and diff
        cmd = [
            "git",
            "-C", str(repo_path),
            "log",
            f"-{limit}",
            "--format=%H%n%an%n%ae%n%ai%n%s%n%b%n---COMMIT-END---"
        ]

        if since:
            cmd.append(f"--since={since}")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git log failed: {e.stderr}")

        commits = []
        current_commit_lines = []

        for line in result.stdout.split('\n'):
            if line == '---COMMIT-END---':
                if current_commit_lines:
                    commits.append(self._parse_commit_lines(current_commit_lines))
                    current_commit_lines = []
            else:
                current_commit_lines.append(line)

        return commits

    def _parse_commit_lines(self, lines: List[str]) -> Dict:
        """Parse commit info from git log lines."""
        if len(lines) < 5:
            return {}

        sha = lines[0]
        author = lines[1]
        email = lines[2]
        date = lines[3]
        subject = lines[4]
        body = '\n'.join(lines[5:]) if len(lines) > 5 else ''

        return {
            'sha': sha,
            'author': author,
            'email': email,
            'date': date,
            'subject': subject,
            'body': body.strip()
        }

    def _get_commit_diff(self, repo_path: Path, sha: str) -> str:
        """Get diff for a commit."""
        cmd = ["git", "-C", str(repo_path), "show", "--format=", sha]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError:
            return ""

    def index_repository(
        self,
        repo_path: str,
        limit: int = 100,
        since: Optional[str] = None,
        include_diff: bool = True,
        verbose: bool = False
    ) -> Dict[str, int]:
        """
        Index git commits from a repository.

        Args:
            repo_path: Path to git repository
            limit: Maximum number of commits to index
            since: Only index commits since this date
            include_diff: Whether to include diff in indexed content
            verbose: Show progress

        Returns:
            Stats dict with 'added', 'skipped', 'failed'
        """
        repo = Path(repo_path).resolve()
        if not (repo / ".git").exists():
            raise FileNotFoundError(f"Not a git repository: {repo}")

        stats = {'added': 0, 'skipped': 0, 'failed': 0}
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3

        if verbose:
            print(f"Fetching commits from {repo}...", file=sys.stderr)

        commits = self._get_commits(repo, limit=limit, since=since)

        if verbose:
            print(f"Found {len(commits)} commits to process...", file=sys.stderr)

        for i, commit in enumerate(commits):
            if not commit or 'sha' not in commit:
                stats['failed'] += 1
                continue

            sha = commit['sha']

            # Skip if already indexed
            if sha in self.indexed_commits:
                stats['skipped'] += 1
                if verbose:
                    print(f"\r[{i+1}/{len(commits)}] Skipped {sha[:8]} (already indexed)", end="", flush=True, file=sys.stderr)
                continue

            try:
                if verbose:
                    print(f"\r[{i+1}/{len(commits)}] Indexing {sha[:8]}: {commit['subject'][:50]}...", end="", flush=True, file=sys.stderr)

                # Build searchable text
                text_parts = [
                    commit['subject'],
                    commit['body']
                ]

                if include_diff:
                    diff = self._get_commit_diff(repo, sha)
                    # Limit diff size to avoid embedding errors (max ~8000 chars)
                    if len(diff) > 8000:
                        diff = diff[:8000] + "\n... (diff truncated)"
                    text_parts.append(diff)

                full_text = '\n\n'.join(filter(None, text_parts))

                # Limit total text size for embedding
                if len(full_text) > 10000:
                    full_text = full_text[:10000] + "\n... (content truncated)"

                # Add to index
                self.rag_db.add_file(
                    file_path=sha,
                    content=full_text,
                    meta={
                        'sha': sha,
                        'short_sha': sha[:8],
                        'author': commit['author'],
                        'email': commit['email'],
                        'date': commit['date'],
                        'subject': commit['subject'],
                        'body': commit['body'],
                        'type': 'commit'
                    }
                )

                self.indexed_commits[sha] = commit['date']
                stats['added'] += 1
                consecutive_failures = 0  # Reset on success

            except Exception as e:
                stats['failed'] += 1
                consecutive_failures += 1

                if verbose:
                    print(f"\nError indexing {sha[:8]}: {e}", file=sys.stderr)

                # Stop if we hit too many consecutive failures (likely Ollama issue)
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    error_msg = f"\nStopped after {consecutive_failures} consecutive embedding failures.\n"
                    if "500" in str(e) or "Internal Server Error" in str(e):
                        error_msg += "Ollama returned HTTP 500 errors. This usually means:\n"
                        error_msg += "  - Text is too long for embedding (try --no-diff mode)\n"
                        error_msg += "  - Ollama model is having issues (try: ollama pull nomic-embed-text)\n"
                    elif "Connection" in str(e) or "refused" in str(e):
                        error_msg += "Cannot connect to Ollama. Is it running? (ollama serve)\n"
                    else:
                        error_msg += f"Error: {e}\n"

                    print(error_msg, file=sys.stderr)
                    break

        # Save state
        self.rag_db.persist()
        self._save_indexed()

        if verbose:
            print(f"\n\nDone. Added: {stats['added']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}", file=sys.stderr)

        return stats
