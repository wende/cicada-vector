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

**Using uvx (Recommended):**
```bash
# Start the server (stdio mode)
export CICADA_HYBRID_DIR=./my_db
uvx --from cicada-vector cicada-vec-server
```

**Using pip:**
```bash
pip install 'cicada-vector[server]'
export CICADA_HYBRID_HYBRID_DIR=./my_db
cicada-vec-server
```

**Tools:** `search_vectors`, `search_hybrid`, `search_code_context`, `index_directory`.

## Quick Start

```python
from cicada_vector import HybridDB

# Initialize
db = HybridDB("./my_knowledge_base")

# Add data (get vector from Ollama/OpenAI)
db.add(id="auth.py", vector=[...], text="def login()...", meta={"path": "..."})

# Search with confidence scores
results = db.search(query_text="login", query_vector=[...], k=5)
for id, score, meta in results:
    print(f"[{score:.4f}] Found {id}")
```

## The Stack

*   **Brains:** [Ollama](https://ollama.com/) (Recommended: `nomic-embed-text`)
*   **Storage:** JSONL (Human-readable, append-only)
*   **Engine:** Pure Python (with optional Numpy acceleration)

---
*Part of the Cicada suite. Simple, effective code intelligence.*