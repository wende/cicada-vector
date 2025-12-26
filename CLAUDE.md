# Cicada Vector - AI Assistant Guidelines

## Project Overview

Cicada Vector is a lightweight semantic search engine for text files with zero dependencies (beyond Ollama for embeddings). It provides fast, incremental indexing with JSONL-based storage and implements a hybrid search combining vector embeddings with keyword matching. The project includes three main CLI tools: `cigrep` for semantic file search (like grep), `cilog` for semantic git commit search, and `cicada-vec` for general indexing and search operations.

## Architecture: Two-Level RAG

The system implements a **two-level RAG (Retrieval-Augmented Generation)** architecture:

1. **File-level (Broad Search)**: Indexes entire files with file-level embeddings (truncated to ~900 chars) plus full-text keyword indexing. This quickly identifies relevant files.

2. **Chunk-level (Specific Search)**: For matched files, creates chunks on-demand during search:
   - **Keyword matches**: Uses 3-line sliding windows to find all occurrences of query terms
   - **Semantic matches**: Uses 15-line chunks (stride 15) with lazy embedding creation and caching

Chunks are **created on-demand** only for files that match the broad search, then cached for future searches. This makes indexing fast (only file-level) while keeping search results precise.

## Vector Database Format

### Input Format: Indexing
Files are indexed in two stages:

**Stage 1 - File-level (at index time):**
```
file_path -> {
  id: "path/to/file.py"
  vector: [768-dim embedding of truncated content]
  metadata: {
    content: "full file text...",
    file_path: "path/to/file.py"
  }
}
```

**Stage 2 - Chunk-level (lazy, at search time):**
```
file_path#chunk_<offset> -> {
  id: "path/to/file.py#chunk_45"     # offset = 0-based line index where chunk starts
  vector: [768-dim embedding of chunk]
  metadata: {
    file_path: "path/to/file.py",
    start_line: 46,                    # offset + 1 (1-based)
    end_line: 60,                      # start_line + 15
    chunk_text: "line 46\nline 47\n...\nline 60",
    is_chunk: true
  }
}
```

Chunks use format `file_path#chunk_{i}` where `i` is the 0-based line offset (e.g., `chunk_45` means lines 46-60). Chunks are 15-line windows with stride 15 (non-overlapping), created only for files that match the broad search, then cached.

### Output Format: Retrieval
Search returns results with semantic meaning:

```python
search("embedding vectors") -> [
  {
    "file": "src/cicada_vector/hybrid.py",
    "score": 0.87,           # Cosine similarity (0.0-1.0)
    "line": 32,              # Start line of chunk
    "snippet": "line 32\nline 33\n...\nline 46",  # 15-line chunk or 3-line window
    "full_match": true       # true=keyword match, false=semantic only
  },
  ...
]
```

The `line` field indicates where the chunk starts in the file. For keyword matches, it's the start of a 3-line window. For semantic matches, it's the start of a 15-line chunk.

## Core Components

### `src/cicada_vector/rag.py` - VectorIndex
The main RAG implementation. Key methods:
- `add_file()`: Indexes files at file-level only (no chunks yet)
- `search()`: Two-level search with chunk filtering (`#chunk_` IDs are filtered from broad search)
- `_find_all_windows()`: Finds 3-line keyword match windows, deduplicated by overlap
- `_find_best_chunks_for_file()`: Creates/loads 15-line chunks, returns best scoring chunk

Returns: `List[Dict]` with `file`, `score`, `line`, `snippet` (chunk text), `full_match` (bool)

### `src/cicada_vector/hybrid.py` - Store
Hybrid search combining vector + keyword:
- Uses `EmbeddingDB` (vectors.jsonl) and `KeywordDB` (keywords.json)
- `_score_boosting_merge()`: Vector scores are boosted by keyword matches (0.2x keyword score)
- Keyword-only matches get 0.4-0.6 range scores (lower than semantic+keyword combos)

### `src/cicada_vector/cigrep.py` - Semantic grep
CLI tool that mimics ripgrep's output format:
- Groups results by file (filename header once, then all matches)
- For keyword matches: shows all matching windows
- For semantic matches: finds the best line within the chunk for display
- Supports `-A`, `-B`, `-C` context flags (reads files to show surrounding lines)
- Supports `--files-only` for piping to other tools
- Uses ANSI colors: magenta for filenames, green for line numbers, cyan for context

### Storage Structure

```
~/.cicada/cigrep/<project_hash>/
├── vectors.jsonl       # Vector embeddings (file-level + cached chunks)
└── keywords.json       # Keyword index (full text)

~/.cicada/cilog/<repo_hash>/
├── vectors.jsonl       # Commit embeddings
└── keywords.json       # Commit message keywords
```

Chunk IDs use format: `{file_path}#chunk_{line_offset}` to enable caching.

## Key Implementation Details

### Chunk Caching (lines 169-219 in rag.py)
- Checks if chunk exists: `chunk_id in self.db.vector_db.ids`
- If not, creates embedding with context: `f"File: {file_path} (lines {start}-{end})\n\n{chunk_text}"`
- Stores with metadata: `is_chunk=True`, `start_line`, `end_line`, `chunk_text`
- Persists immediately after creating new chunks

### Keyword Match Deduplication (lines 128-145 in rag.py)
Windows are deduplicated by overlap: two windows overlap if `abs(line_idx - used_start) < window_size`. Keeps higher-scoring window.

### Cosine Similarity
Computed inline (lines 226-234) without external dependencies:
```python
dot = sum(q * v for q, v in zip(query_vector, chunk_vector))
similarity = dot / (sqrt(sum(q*q)) * sqrt(sum(v*v)))
```

### Display Logic in cigrep (lines 218-239)
For chunks, finds the line with most query terms:
```python
for idx, line in enumerate(snippet_lines):
    score = sum(1 for term in query_terms if term in line.lower())
matched_line = snippet_lines[best_line_idx]
```

## CLI Tools

### cigrep - Semantic file search
```bash
cigrep "query" [path] -k 5 -C 2 --files-only
```
- Incremental indexing (skips unchanged files)
- Silent mode by default (only shows results)
- `--verbose` shows indexing progress
- `--clean` removes index for directory
- `--no-index` skips indexing (search only)

### cilog - Semantic commit search
```bash
cilog "query" [path] -k 5 --since="2 weeks ago" --limit=100
```
- Indexes git commit messages (with optional diffs via `--with-diff`)
- Shows commit SHA, score, and snippet

### cicada-vec - General purpose
```bash
cicada-vec index <path>
cicada-vec search "query" -k 5
```

## Development

- **Package manager**: Use `uv` for all Python commands
- **Installation**: `uv tool install cicada-vector` or `uv pip install -e .` for development
- **No Makefile**: Run commands directly with `uv run`
- **No test framework installed**: Tests exist in `tests/` but pytest is not in dependencies

## Important Notes

1. **Chunks pollute search results**: The broad search can return cached chunks. Always filter `#chunk_` from doc_id in broad search and fetch `k*10` results to compensate.

2. **Snippet format**: Other consumers (cilog, MCP server) expect `snippet` to be a text chunk (3-15 lines), not a single line. Only cigrep's display layer extracts individual lines.

3. **Ollama embedding limits**: See [docs/OLLAMA_EMBEDDING_LIMITS.md](docs/OLLAMA_EMBEDDING_LIMITS.md) for detailed research on model-specific token limits. Current implementation uses hardcoded 900 char truncation, but this should be replaced with adaptive limit discovery using the `/api/embed` endpoint's `prompt_eval_count`.

4. **No eager chunking**: Never create chunks at index time. Only create on-demand during search for matched files.

5. **Score ranges**:
   - Vector match: cosine similarity (0.0-1.0)
   - Vector + keyword boost: up to 1.0 (capped)
   - Keyword-only: 0.4-0.6 range
