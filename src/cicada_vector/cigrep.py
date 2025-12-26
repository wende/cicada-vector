"""
cigrep: Zero-config Semantic Grep with ripgrep-style output.
Usage: cigrep "query string" [path]
"""

import argparse
import hashlib
import os
import re
import shutil
import sys
from collections import defaultdict
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
    parser = argparse.ArgumentParser(
        description="cigrep: Semantic search for your code.",
        epilog="Examples:\n"
               "  cigrep 'authentication logic'\n"
               "  cigrep 'how do I login' -k 10\n"
               "  cigrep 'database connection' -C 2\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("path", nargs="?", default=".", help="Directory to search (default: .)")
    parser.add_argument("-k", type=int, default=5, help="Number of files to search")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--no-index", action="store_true", help="Skip indexing (search only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show indexing progress")
    parser.add_argument("--clean", action="store_true", help="Remove index for this directory")
    parser.add_argument("--files-only", "-l", action="store_true", help="Only print file paths")
    # Context flags (like grep/rg)
    parser.add_argument("-A", type=int, default=0, metavar="NUM", help="Show NUM lines after match")
    parser.add_argument("-B", type=int, default=0, metavar="NUM", help="Show NUM lines before match")
    parser.add_argument("-C", type=int, default=0, metavar="NUM", help="Show NUM lines before and after match")

    args = parser.parse_args()

    target_path = Path(args.path).resolve()
    if not target_path.exists():
        print(f"Error: Path not found: {target_path}", file=sys.stderr)
        sys.exit(1)

    # Setup storage
    project_hash = get_project_hash(target_path)
    storage_dir = CACHE_DIR / project_hash

    # Handle --clean
    if args.clean:
        if storage_dir.exists():
            shutil.rmtree(storage_dir)
            print(f"Removed index for {target_path}", file=sys.stderr)
        else:
            print(f"No index found for {target_path}", file=sys.stderr)
        return

    # Require query if not cleaning
    if not args.query:
        parser.error("query is required unless --clean is specified")

    if not storage_dir.exists():
        os.makedirs(storage_dir)

    indexer = DirectoryIndexer(
        storage_dir=str(storage_dir),
        ollama_host=DEFAULT_OLLAMA_HOST,
        ollama_model=args.model
    )

    # 1. Incremental Index (unless skipped)
    indexed_count = 0
    if not args.no_index:
        if args.verbose:
            print(f"Indexing {target_path}...", file=sys.stderr)

        try:
            stats = indexer.index_directory(target_path, verbose=args.verbose)
            indexed_count = stats['added']
            if args.verbose:
                print(f"Index updated: +{stats['added']} files, {stats['skipped']} skipped.", file=sys.stderr)
        except Exception as e:
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
        print("No matches found.")
        return

    # Handle context lines
    context_before = args.B if args.B > 0 else args.C
    context_after = args.A if args.A > 0 else args.C

    # Extract query terms for highlighting
    query_terms = [t.lower() for t in re.findall(r"\w+", args.query)]

    try:
        if args.files_only:
            # Just print unique file paths
            seen = set()
            for res in results:
                rel_path = res['file']
                try:
                    abs_path = target_path / rel_path
                    display_path = os.path.relpath(abs_path, Path.cwd())
                except:
                    display_path = rel_path

                if display_path not in seen:
                    seen.add(display_path)
                    print(display_path)
        elif context_before > 0 or context_after > 0:
            # Show context lines (need to read file)
            grouped = defaultdict(list)
            for res in results:
                grouped[res['file']].append(res)

            for file_idx, (rel_path, file_results) in enumerate(grouped.items()):
                try:
                    abs_path = target_path / rel_path
                    display_path = os.path.relpath(abs_path, Path.cwd())
                except:
                    display_path = rel_path
                    abs_path = Path(rel_path)

                # Read file for context
                try:
                    with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                        file_lines = f.read().splitlines()
                except:
                    file_lines = []

                # Print file header
                print(f"\033[35m{display_path}\033[0m")

                # Track which lines we've already shown to avoid duplicates
                shown_ranges = []

                for res in file_results:
                    snippet = res['snippet'].strip()
                    snippet_lines = snippet.split('\n')

                    # Find best line in snippet
                    best_score = 0
                    best_line_idx = 0
                    for idx, line in enumerate(snippet_lines):
                        line_lower = line.lower()
                        score = sum(1 for term in query_terms if term in line_lower)
                        if score > best_score:
                            best_score = score
                            best_line_idx = idx

                    matched_line_num = res['line'] + best_line_idx

                    # Calculate context range
                    start_line = max(1, matched_line_num - context_before)
                    end_line = min(len(file_lines), matched_line_num + context_after)

                    # Skip if this overlaps with already shown range
                    overlaps = any(
                        not (end_line < shown_start or start_line > shown_end)
                        for shown_start, shown_end in shown_ranges
                    )
                    if overlaps:
                        continue

                    shown_ranges.append((start_line, end_line))

                    # Print separator if not first match in file
                    if len(shown_ranges) > 1:
                        print("\033[36m--\033[0m")

                    # Print context lines
                    for line_num in range(start_line, end_line + 1):
                        if line_num <= len(file_lines):
                            line_content = file_lines[line_num - 1]
                            if line_num == matched_line_num:
                                print(f"\033[32m{line_num}\033[0m:{line_content} \033[90m[{res['score']:.2f}]\033[0m")
                            else:
                                print(f"\033[32m{line_num}\033[0m\033[36m-\033[0m{line_content}")

                # Blank line between files
                if file_idx < len(grouped) - 1:
                    print()
        else:
            # Normal output (ripgrep-style, grouped by file)
            grouped = defaultdict(list)
            for res in results:
                grouped[res['file']].append(res)

            for file_idx, (rel_path, file_results) in enumerate(grouped.items()):
                try:
                    abs_path = target_path / rel_path
                    display_path = os.path.relpath(abs_path, Path.cwd())
                except:
                    display_path = rel_path

                # Print file header (magenta like rg)
                print(f"\033[35m{display_path}\033[0m")

                # Print each match
                for res in file_results:
                    snippet = res['snippet'].strip()
                    if snippet:
                        snippet_lines = snippet.split('\n')
                        # Find the line with the most query terms
                        best_score = 0
                        best_line_idx = 0
                        for idx, line in enumerate(snippet_lines):
                            line_lower = line.lower()
                            score = sum(1 for term in query_terms if term in line_lower)
                            if score > best_score:
                                best_score = score
                                best_line_idx = idx

                        matched_line = snippet_lines[best_line_idx]
                        matched_line_num = res['line'] + best_line_idx
                    else:
                        matched_line = ""
                        matched_line_num = res['line']

                    print(f"\033[32m{matched_line_num}\033[0m:{matched_line} \033[90m[{res['score']:.2f}]\033[0m")

                # Blank line between files
                if file_idx < len(grouped) - 1:
                    print()

        # Print index status at the end if files were indexed
        if indexed_count > 0:
            print(f"Indexed {indexed_count} new files.", file=sys.stderr)

    except BrokenPipeError:
        # Handle pipe being closed (e.g., piping to head)
        sys.stderr.close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.stderr.close()
