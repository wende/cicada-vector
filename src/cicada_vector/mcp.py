"""
MCP Server for Cicada Vector.
Exposes vector search capabilities to AI assistants.
"""

import os
import json
import urllib.request
from pathlib import Path
from typing import List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    msg = (
        "MCP server dependencies not installed.\n"
        "  - If using pip: pip install 'cicada-vector[server]'\n"
        "  - If using uv run: uv run --extra server cicada-vec-server\n"
        "  - If using uvx: uvx --from 'cicada-vector[server]' cicada-vec-server"
    )
    raise ImportError(msg)

from .db import EmbeddingDB
from .rag import VectorIndex
from .indexer import DirectoryIndexer

# Configuration
DB_PATH = os.environ.get("CICADA_DB_PATH", "cicada_db")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")

# Initialize MCP
mcp = FastMCP("cicada-vector")

def get_embedding(text: str) -> List[float]:
    """Helper to get embedding from Ollama."""
    url = f"{OLLAMA_HOST}/api/embeddings"
    data = {"model": OLLAMA_MODEL, "prompt": text}
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        return result["embedding"]

@mcp.tool()
def index_directory(path: str) -> str:
    """
    Incrementally index a directory into the database.
    
    Args:
        path: Absolute path to the directory to index.
    """
    try:
        indexer = DirectoryIndexer(
            DB_PATH,
            ollama_host=OLLAMA_HOST,
            ollama_model=OLLAMA_MODEL
        )
        stats = indexer.index_directory(path)
        return f"Indexing complete: {stats['added']} added, {stats['skipped']} skipped, {stats['failed']} failed."
    except Exception as e:
        return f"Indexing failed: {e}"

@mcp.tool()
def search_vectors(query: str, k: int = 5) -> str:
    """
    Search for similar items using pure vector semantic search.
    Useful for finding concepts, synonyms, or related ideas.
    
    Args:
        query: The semantic query (e.g. "authentication logic")
        k: Number of results to return (default 5)
    """
    vectors_path = os.path.join(DB_PATH, "vectors.jsonl")
    if not os.path.exists(vectors_path):
        return f"Error: Vector DB not found at {vectors_path}. Run index_directory first."

    try:
        db = EmbeddingDB(vectors_path)
        query_vec = get_embedding(query)
        results = db.search(query_vec, k=k)
        
        output = []
        for id, score, meta in results:
            text = meta.get("text", str(meta))
            output.append(f"[{score:.4f}] {id}\n{text}\n")
            
        return "\n".join(output)
    except Exception as e:
        return f"Search failed: {e}"

@mcp.tool()
def search(query: str, k: int = 5) -> str:
    """
    Two-stage hybrid search (vector + keyword) that returns specific paragraphs.
    Stage 1: finds the most relevant files using hybrid vector+keyword search.
    Stage 2: within those files, returns the specific 15-line chunks that best match.

    Args:
        query: The question or topic (e.g. "how to connect to db")
        k: Number of paragraph snippets to return
    """
    if not os.path.exists(DB_PATH):
        return f"Error: Hybrid/RAG DB directory not found at {DB_PATH}"
        
    try:
        db = VectorIndex(DB_PATH)
        query_vec = get_embedding(query)
        results = db.search(query, query_vec, k=k)
        
        output = []
        for res in results:
            output.append(f"File: {res['file']} (Line {res['line']})")
            output.append(f"Confidence: {res['score']:.4f}")
            output.append("```")
            output.append(res['snippet'])
            output.append("```\n")
            
        return "\n".join(output)
    except Exception as e:
        return f"RAG search failed: {e}"

def main():
    mcp.run()

if __name__ == "__main__":
    main()
