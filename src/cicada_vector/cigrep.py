"""
cigrep: Zero-config Semantic File Search.
Usage: cigrep "query string" [path]
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

from cicada_vector.indexer import DirectoryIndexer
from cicada_vector.rag import VectorIndex

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")
CACHE_DIR = Path.home() / ".cicada" / "cigrep"

def get_project_hash(path: Path) -> str:
    """Generate a stable hash for the project path."""
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="cigrep: Semantic search for text files (code, docs, configs, etc.)")
    parser.add_argument("query", help="Search query (e.g., 'auth logic', 'installation steps', 'database config')")
    parser.add_argument("path", nargs="?", default=".", help="Directory to search (default: .)")
    parser.add_argument("-k", type=int, default=5, help="Number of results")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--no-index", action="store_true", help="Skip indexing (search only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show indexing progress")
    
    args = parser.parse_args()
    
    target_path = Path(args.path).resolve()
    if not target_path.exists():
        print(f"Error: Path not found: {target_path}")
        sys.exit(1)
        
    # Setup storage
    project_hash = get_project_hash(target_path)
    storage_dir = CACHE_DIR / project_hash
    
    if not storage_dir.exists():
        os.makedirs(storage_dir)
        
    indexer = DirectoryIndexer(
        storage_dir=str(storage_dir),
        ollama_host=DEFAULT_OLLAMA_HOST,
        ollama_model=args.model
    )
    
    # 1. Incremental Index (unless skipped)
    if not args.no_index:
        # Always show we're indexing (even if not verbose)
        print(f"Indexing {target_path}...", file=sys.stderr, end='', flush=True)

        try:
            stats = indexer.index_directory(target_path, verbose=args.verbose)

            # Clear the "Indexing..." line if not verbose
            if not args.verbose:
                print(f"\r", end='', file=sys.stderr)

            # Always show result (brief if not verbose)
            if args.verbose:
                print(f"Index updated: +{stats['added']} files, {stats['skipped']} skipped, {stats['failed']} failed.", file=sys.stderr)
            elif stats['added'] > 0:
                print(f"Indexed {stats['added']} new files.", file=sys.stderr)
            # If 0 added, don't print anything (already up to date, silent)

        except Exception as e:
            # Clear the "Indexing..." line
            print(f"\r", end='', file=sys.stderr)
            # If Ollama fails, we might still want to search existing index
            print(f"Warning: Indexing failed ({e}). Searching existing data...", file=sys.stderr)

    # 2. Search
    db = VectorIndex(str(storage_dir), ollama_host=DEFAULT_OLLAMA_HOST, ollama_model=args.model)

    # Search (embedding handled internally by VectorIndex)
    try:
        results = db.search(args.query, k=args.k)
    except Exception as e:
        print(f"Error: Could not search: {e}", file=sys.stderr)
        print("Is Ollama running?", file=sys.stderr)
        sys.exit(1)
    
    if not results:
        print("No matches found.")
        return

    print(f"\nFound {len(results)} matches for '{args.query}':\n")
    
    for res in results:
        rel_path = res['file']
        # Try to show path relative to current dir for copy-pasteability
        try:
            # We stored relative paths in the DB relative to project root
            # Reconstruct absolute
            abs_path = target_path / rel_path
            display_path = os.path.relpath(abs_path, Path.cwd())
        except:
            display_path = rel_path
            
        print(f"\033[1m{display_path}\033[0m:{res['line']}  \033[90m[{res['score']:.2f}]\033[0m")
        
        # Indent snippet
        snippet = res['snippet'].strip()
        for line in snippet.splitlines():
            print(f"   {line}")
        print()

if __name__ == "__main__":
    main()
