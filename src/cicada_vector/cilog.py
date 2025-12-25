"""
cilog: Semantic Git Commit Search.
Usage: cilog "query string" [repo-path]
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

from cicada_vector.git_indexer import GitIndexer
from cicada_vector.rag import VectorIndex

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
CACHE_DIR = Path.home() / ".cicada" / "cilog"


def get_repo_hash(path: Path) -> str:
    """Generate a stable hash for the repository path."""
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="cilog: Semantic search for git commits.",
        epilog="Examples:\n"
               "  cilog 'authentication bug fix'\n"
               "  cilog 'refactor API' --limit 500\n"
               "  cilog 'performance' --since '1 month ago'\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", help="Search query (e.g., 'auth bug fix', 'refactor API')")
    parser.add_argument("path", nargs="?", default=".", help="Git repository path (default: .)")
    parser.add_argument("-k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--limit", type=int, default=100, help="Max commits to index (default: 100)")
    parser.add_argument("--since", help="Index commits since date (e.g., '2024-01-01', '7 days ago')")
    parser.add_argument("--with-diff", action="store_true", help="Include diffs in index (slower, may fail on large commits)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--no-index", action="store_true", help="Skip indexing (search only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show indexing progress")
    parser.add_argument("--show-diff", action="store_true", help="Show commit diff in results")

    args = parser.parse_args()

    repo_path = Path(args.path).resolve()
    if not repo_path.exists():
        print(f"Error: Path not found: {repo_path}", file=sys.stderr)
        sys.exit(1)

    if not (repo_path / ".git").exists():
        print(f"Error: Not a git repository: {repo_path}", file=sys.stderr)
        sys.exit(1)

    # Setup storage
    repo_hash = get_repo_hash(repo_path)
    storage_dir = CACHE_DIR / repo_hash

    if not storage_dir.exists():
        os.makedirs(storage_dir)

    indexer = GitIndexer(
        storage_dir=str(storage_dir),
        ollama_host=DEFAULT_OLLAMA_HOST,
        ollama_model=args.model
    )

    # 1. Index commits (unless skipped)
    if not args.no_index:
        # Always show we're indexing (even if not verbose)
        print(f"Indexing commits from {repo_path}...", file=sys.stderr, end='', flush=True)

        try:
            stats = indexer.index_repository(
                repo_path=str(repo_path),
                limit=args.limit,
                since=args.since,
                include_diff=args.with_diff,
                verbose=args.verbose
            )

            # Clear the "Indexing..." line if not verbose
            if not args.verbose:
                print(f"\r", end='', file=sys.stderr)

            # Always show result (brief if not verbose)
            if args.verbose:
                print(f"Index updated: +{stats['added']} commits, {stats['skipped']} skipped, {stats['failed']} failed.", file=sys.stderr)
            elif stats['added'] > 0:
                print(f"Indexed {stats['added']} new commits.", file=sys.stderr)
            # If 0 added, don't print anything (already up to date, silent)

        except Exception as e:
            # Clear the "Indexing..." line
            print(f"\r", end='', file=sys.stderr)
            print(f"Warning: Indexing failed ({e}). Searching existing data...", file=sys.stderr)

    # 2. Search
    db = VectorIndex(str(storage_dir), ollama_host=DEFAULT_OLLAMA_HOST, ollama_model=args.model)

    try:
        results = db.search(args.query, k=args.k)
    except Exception as e:
        print(f"Error: Search failed: {e}", file=sys.stderr)
        print("Is Ollama running?", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No commits found matching your query.")
        return

    print(f"\nFound {len(results)} commits matching '{args.query}':\n")

    for res in results:
        sha = res['file']
        score = res['score']

        # Get metadata from the underlying store
        # The VectorIndex wraps Store, and we can access via search results
        # But for cleaner access, let's get it from the snippet metadata

        print(f"\033[1;33m{sha[:8]}\033[0m  \033[90m[{score:.2f}]\033[0m")

        # Show the relevant snippet
        snippet = res['snippet'].strip()
        if snippet:
            for line in snippet.splitlines()[:10]:  # Limit to first 10 lines
                print(f"  {line}")

        print()

    print(f"\nTip: Use 'git show <sha>' to see full commit details")


if __name__ == "__main__":
    main()
