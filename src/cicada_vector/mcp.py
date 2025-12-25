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
    raise ImportError("MCP server dependencies not installed. Run: pip install 'cicada-vector[server]'" )

from .db import VectorDB
from .hybrid import HybridDB
from .rag import RagDB

# Configuration
DB_PATH = os.environ.get("CICADA_VECTOR_DB", "vectors.jsonl")
HYBRID_DIR = os.environ.get("CICADA_HYBRID_DIR", "hybrid_db")
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
def search_vectors(query: str, k: int = 5) -> str:
    """
    Search for similar items using pure vector semantic search.
    Useful for finding concepts, synonyms, or related ideas.
    
    Args:
        query: The semantic query (e.g. "authentication logic")
        k: Number of results to return (default 5)
    """
    if not os.path.exists(DB_PATH):
        return f"Error: Vector DB not found at {DB_PATH}"
        
    try:
        db = VectorDB(DB_PATH)
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
def search_hybrid(query: str, k: int = 5) -> str:
    """
    Search using Hybrid (Vector + Keyword) strategy.
    BEST for code search where queries might contain exact identifiers (e.g. "UserAuth")
    or general concepts (e.g. "login").
    
    Args:
        query: The query text
        k: Number of results
    """
    if not os.path.exists(HYBRID_DIR):
        return f"Error: Hybrid DB directory not found at {HYBRID_DIR}"
        
    try:
        db = HybridDB(HYBRID_DIR)
        query_vec = get_embedding(query)
        # Pass query text for keyword search, vector for semantic
        results = db.search(query, query_vec, k=k)
        
        output = []
        for id, score, meta in results:
            text = meta.get("text", str(meta))
            output.append(f"[{score:.4f}] {id}\n{text}\n")
            
        return "\n".join(output)
    except Exception as e:
        return f"Hybrid search failed: {e}"

@mcp.tool()
def search_code_context(query: str, k: int = 3) -> str:
    """
    Find specific code snippets relevant to the query.
    Uses RAG (Retrieval Augmented Generation) to find the file,
    then scans for the most relevant lines.
    
    Args:
        query: The question or topic (e.g. "how to connect to db")
        k: Number of snippets to return
    """
    if not os.path.exists(HYBRID_DIR):
        return f"Error: Hybrid/RAG DB directory not found at {HYBRID_DIR}"
        
    try:
        db = RagDB(HYBRID_DIR)
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
