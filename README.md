# Cicada Vector 🦗

**A lightweight semantic search engine for developers.**

Cicada Vector is a simple, zero-dependency semantic search engine and RAG database. It explores a different approach to code intelligence: maximizing semantic awareness while minimizing complexity and dependencies.

## Why this exists?

The original Cicada is powerful because it deeply *understands* code structure (SCIP, ASTs). However, that power often requires heavy dependencies and longer setup times.

**Cicada Vector takes a complementary path:**
It focuses on **Semantic Awareness** and ease of use. By combining local LLM embeddings (via Ollama) with a hybrid database, it provides robust search capabilities with a minimal footprint.

## Features

*   **Lightweight:** Minimal Python codebase. **Zero dependencies** (Standard Library only) for the core engine.
*   **Instant Install:** No waiting for heavy ML libraries to compile.
*   **Semantic Intelligence:** Understands *intent*. Searching for "auth" finds login logic, even if the word "auth" isn't present.
*   **Hybrid Search:** Combines Vector semantic search with Keyword exact matching. It won't miss specific identifiers like `UserAuth_v2`.
*   **Simple RAG:** A "Search Broad -> Scan Specific" pipeline that pinpoints relevant code snippets.
*   **MCP Ready:** Built-in Model Context Protocol server for immediate use with AI assistants.

## Tools

### 1. `cigrep` (Zero-config Semantic Search)
The fastest way to search your code semantically. No setup needed.
```bash
cigrep "how do I handle authentication"
```
It automatically indexes changed files in the background and searches instantly.

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

```python
from cicada_vector import Store

# Initialize (embeddings handled automatically via Ollama)
db = Store("./my_knowledge_base")

# Add data - just provide text, we handle embeddings
db.add(id="auth.py", text="def login(username, password):\n    ...", meta={"path": "src/auth.py"})
db.add(id="user.py", text="class User:\n    ...", meta={"path": "src/user.py"})

# Search - just provide query text, we handle embeddings
results = db.search("how to authenticate users", k=5)
for id, score, meta in results:
    print(f"[{score:.4f}] {id}: {meta.get('path')}")

# Custom embedding provider (optional)
from cicada_vector import EmbeddingProvider

class MyCustomEmbedder:
    def embed(self, text: str) -> list[float]:
        # Use OpenAI, HuggingFace, or any embedding service
        return my_embedding_service(text)

db = Store("./my_db", embedding_provider=MyCustomEmbedder())
```

## Indexing Custom Data

Cicada Vector isn't just for code files - index any text data:

```python
from cicada_vector import Store
import subprocess
import json

# Example: Index git commits
db = Store("./git_commits_db")

result = subprocess.run(
    ["git", "log", "--format=%H|%an|%s|%b", "-10"],
    capture_output=True, text=True
)

for line in result.stdout.strip().split('\n'):
    sha, author, subject, body = line.split('|', 3)
    commit_text = f"{subject}\n{body}"

    db.add(
        id=sha,
        text=commit_text,
        meta={"author": author, "subject": subject, "type": "commit"}
    )

# Search commits
results = db.search("authentication bug fix", k=5)
for sha, score, meta in results:
    print(f"[{score:.4f}] {sha[:8]} - {meta['subject']}")
```

**Use cases:**
- Git commits and history
- GitHub PRs and issues
- Documentation sites
- Support tickets
- Any text corpus

## The Stack

*   **Brains:** [Ollama](https://ollama.com/) (Recommended: `nomic-embed-text`)
*   **Storage:** JSONL (Human-readable, append-only)
*   **Engine:** Pure Python (with optional Numpy acceleration)

---
*Part of the Cicada suite. Simple, effective code intelligence.*