# Cicada Vector 🦗

**A lightweight semantic search engine for text files.**

Cicada Vector is a simple, zero-dependency semantic search engine and RAG database. Search any text content semantically - code, documentation, commits, or custom data.

## Why this exists?

The original Cicada is powerful because it deeply *understands* code structure (SCIP, ASTs). However, that power often requires heavy dependencies and longer setup times.

**Cicada Vector takes a complementary path:**
It focuses on **Semantic Awareness** for any text content. By combining local LLM embeddings (via Ollama) with a hybrid database, it provides robust search capabilities with a minimal footprint and maximum flexibility.

## Features

*   **Lightweight:** Minimal Python codebase. **Zero dependencies** (Standard Library only) for the core engine.
*   **Instant Install:** No waiting for heavy ML libraries to compile.
*   **Semantic Intelligence:** Understands *intent*. Searching for "auth" finds login logic, even if the word "auth" isn't present.
*   **Hybrid Search:** Combines Vector semantic search with Keyword exact matching. Won't miss exact terms while understanding meaning.
*   **Simple RAG:** A "Search Broad -> Scan Specific" pipeline that pinpoints relevant content snippets.
*   **Universal:** Works on code, docs, commits, configs - any text content.
*   **MCP Ready:** Built-in Model Context Protocol server for immediate use with AI assistants.

## Database Classes

Cicada Vector provides four database classes:

*   **VectorDB**: Pure semantic vector search
*   **KeywordDB**: Traditional keyword-based search
*   **HybridDB**: Combines vector + keyword search (Recommended)
*   **RagDB**: RAG database for file-based search with line numbers

## Tools

### 1. `cigrep` (Semantic File Search)
Zero-config semantic search for any text files - code, docs, configs, anything.
```bash
cigrep "how do I handle authentication"     # Search code
cigrep "installation steps" docs/           # Search docs
cigrep "database config" .                  # Search everything
```
Automatically indexes changed files in the background and searches instantly.

### 2. `cilog` (Semantic Git Commit Search)
Search your git commit history semantically.
```bash
cilog "authentication bug fix"
cilog "refactor API" --limit 500
cilog "performance improvements" --since "1 month ago"
```
Indexes commit messages for fast semantic search. Use `--no-diff` (recommended) for faster indexing.

## MCP Server

AI assistants can use your local knowledge base directly:

```bash
pip install 'cicada-vector[server]'
export CICADA_HYBRID_DIR=./my_db
cicada-vec-server
```

**Tools:**
*   `search_vectors`: Pure semantic search.
*   `search_hybrid`: Vector + Keyword search (Recommended).
*   `search_code_context`: RAG search returning file snippets with line numbers.
*   `index_directory`: Incrementally index a local directory into the database.

**Configuration:**
If using `uv` or `uvx`, ensure you include the `[server]` extra:
```bash
uv tool install "cicada-vector[server]"
```

For manual configuration (e.g., in Claude Desktop or Gemini), set the command to:
`uvx --from "cicada-vector[server]" cicada-vec-server`
And set the environment variable `CICADA_HYBRID_DIR` to your database path.

## Quick Start

### 1. Generate Embeddings

Cicada Vector requires you to provide embeddings (vectors). Use Ollama to generate them:

```python
import json
import urllib.request

def get_embedding(text, model="nomic-embed-text"):
    """Get embedding from Ollama API"""
    url = "http://localhost:11434/api/embeddings"
    data = json.dumps({"model": model, "prompt": text}).encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result['embedding']
```

### 2. Create Database and Add Data

```python
from cicada_vector import HybridDB

# Initialize HybridDB (combines vector + keyword search)
db = HybridDB("./my_knowledge_base")

# Add data with embeddings
auth_text = "def login(username, password):\n    ..."
auth_vector = get_embedding(auth_text)
db.add(id="auth.py", vector=auth_vector, text=auth_text, meta={"path": "src/auth.py"})

user_text = "class User:\n    ..."
user_vector = get_embedding(user_text)
db.add(id="user.py", vector=user_vector, text=user_text, meta={"path": "src/user.py"})

# Persist to disk (optional - data is written on add, but this rewrites the file)
db.persist()
```

### 3. Search

```python
# Generate embedding for query
query = "how to authenticate users"
query_vector = get_embedding(query)

# Hybrid search (recommended - combines vector + keyword)
results = db.search(query_text=query, query_vector=query_vector, k=5)
for doc_id, score, meta in results:
    print(f"[{score:.4f}] {doc_id}: {meta.get('path')}")
```

## Indexing Custom Data

Cicada Vector isn't just for code files - index any text data:

```python
from cicada_vector import HybridDB
import subprocess

# Example: Index git commits
db = HybridDB("./git_commits_db")

result = subprocess.run(
    ["git", "log", "--format=%H|%an|%s|%b", "-10"],
    capture_output=True, text=True
)

for line in result.stdout.strip().split('\n'):
    sha, author, subject, body = line.split('|', 3)
    commit_text = f"{subject}\n{body}"
    commit_vector = get_embedding(commit_text)

    db.add(
        id=sha,
        vector=commit_vector,
        text=commit_text,
        meta={"author": author, "subject": subject, "type": "commit"}
    )

# Search commits
query = "authentication bug fix"
query_vector = get_embedding(query)
results = db.search(query_text=query, query_vector=query_vector, k=5)
for sha, score, meta in results:
    print(f"[{score:.4f}] {sha[:8]} - {meta['subject']}")
```

## Other Database Classes

### VectorDB (Pure Semantic Search)

```python
from cicada_vector import VectorDB

db = VectorDB("./my_vectors.jsonl")

# Add vectors (no text storage)
db.add(id="doc1", vector=get_embedding("some text"), meta={"path": "doc1.txt"})

# Search (reuses get_embedding() helper from Quick Start)
query_vector = get_embedding("search query")
results = db.search(query=query_vector, k=5)
```

### KeywordDB (Traditional Search)

```python
from cicada_vector import KeywordDB

db = KeywordDB("./my_keywords.jsonl")

# Add documents
db.add(id="doc1", text="some text to index")

# Search (OR search - matches any word)
results = db.search(query="search terms")
```

### RagDB (File-based RAG)

```python
from cicada_vector import RagDB

db = RagDB("./my_rag_db")

# Add files
file_content = open("src/auth.py").read()
# Reuse get_embedding() helper from Quick Start
file_vector = get_embedding(file_content)
db.add_file(file_path="src/auth.py", content=file_content, vector=file_vector)

# Search (returns file + line numbers)
query_vector = get_embedding("authentication")
results = db.search(query="authentication", k=3, query_vector=query_vector)
```

**Use cases:**
- Code files (semantic search across your codebase)
- Git commits and history
- GitHub PRs and issues
- Documentation sites
- Configuration files
- Support tickets
- Any text corpus

## The Stack

*   **Brains:** [Ollama](https://ollama.com/) (Recommended: `nomic-embed-text`)
*   **Storage:** JSONL (Human-readable, append-only)
*   **Engine:** Pure Python (with optional Numpy acceleration)

---
*Part of the Cicada suite. Simple, effective semantic search for text.*
