# Cicada Vector

A "poor man's" vector database designed for absolute simplicity, zero dependencies, and developer-centric code exploration.

## Why reinvent the wheel?

Most vector databases (Chroma, Milvus, Qdrant) are built for "Billion-Scale" problems. They come with heavy dependency trees (Torch, ONNX, FastAPI, Pydantic) and complex indexing graphs (HNSW).

**Cicada Vector is built on a different set of assumptions:**

1.  **Scale is Finite:** A typical developer codebase has 1,000 to 100,000 symbols. At this scale, brute-force matrix multiplication is faster than the overhead of traversing complex graph indices.
2.  **Dependencies are Bloat:** Developers shouldn't have to wait 5 minutes for `torch` to compile just to run a local code search tool. Cicada Vector has **zero dependencies** and runs on pure Python standard library.
3.  **Local First:** It uses a simple, append-only JSONL format. You can open your database in a text editor, debug it, and move it around as a single file. No server required.
4.  **Hybrid Acceleration:** While it works on pure Python out of the box, it will automatically detect and use `numpy` for a 50x speed boost if it's available in the environment.

## Features

*   **Zero Dependencies:** Standard Library only.
*   **JSONL Storage:** Human-readable, append-only persistence.
*   **Progressive Engine:** Pure Python loops by default, automatic Numpy acceleration if present.
*   **Simple API:** `add()`, `search()`, `persist()`.

## Quick Start

```python
from cicada_vector import VectorDB

# Initialize
db = VectorDB("my_vectors.jsonl")

# Add a vector (e.g. from Ollama)
db.add("doc_1", [0.1, 0.2, 0.3], {"text": "Hello world"})

# Semantic search
results = db.search([0.1, 0.2, 0.2], k=1)
for id, score, meta in results:
    print(f"Found {id} with score {score}")
```

## End-to-End with Ollama

Cicada Vector is the perfect companion for [Ollama](https://ollama.com/). Combine them to build a full semantic search engine with zero `pip install` wait times.

See `test_e2e_ollama.py` for a full example.
