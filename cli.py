"""
Cicada Vector CLI
Simple interface to index files and search vectors using Ollama.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cicada_vector import VectorDB

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "nomic-embed-text"

class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model

    def check_connection(self) -> bool:
        try:
            url = f"{self.host}/api/tags"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                return response.status == 200
        except Exception:
            return False

    def get_embedding(self, text: str) -> List[float]:
        url = f"{self.host}/api/embeddings"
        data = {"model": self.model, "prompt": text}
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                return result["embedding"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama Error ({e.code}): {e.reason}")
        except Exception as e:
            raise RuntimeError(f"Connection Error: {e}")

def handle_index(args):
    """Index a file (CSV or JSONL)."""
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    # Clean start if requested or if file exists
    if os.path.exists(args.db):
        print(f"Appending to existing DB: {args.db}")
    else:
        print(f"Creating new DB: {args.db}")

    embedder = OllamaEmbedder(DEFAULT_OLLAMA_HOST, args.model)
    if not embedder.check_connection():
        print(f"Error: Could not connect to Ollama at {DEFAULT_OLLAMA_HOST}")
        sys.exit(1)

    db = VectorDB(args.db)
    rows = []

    print(f"Reading {args.file}...")
    
    # Simple parser logic
    if args.file.endswith('.csv'):
        with open(args.file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                # Naive text representation: key: value, key: value
                text = ", ".join([f"{k}: {v}" for k, v in row.items() if v])
                rows.append({"id": f"row_{i}", "text": text, "meta": row})
    elif args.file.endswith('.jsonl'):
        with open(args.file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                data = json.loads(line)
                # Flexible text extraction
                text = data.get("text", data.get("content", str(data)))
                rows.append({"id": f"row_{i}", "text": text, "meta": data})
    else:
        print("Error: Unsupported file format. Please use .csv or .jsonl")
        sys.exit(1)

    print(f"Embedding {len(rows)} items using model '{args.model}'...")
    start_time = time.time()
    
    for i, item in enumerate(rows):
        if i % 10 == 0:
            print(f"\r[{i}/{len(rows)}] {(i/len(rows))*100:.1f}%", end="", flush=True)
        
        try:
            vector = embedder.get_embedding(item['text'])
            db.add(item['id'], vector, item['meta'])
        except Exception as e:
            print(f"\nSkipping row {i}: {e}")

    duration = time.time() - start_time
    rate = len(rows) / duration if duration > 0 else 0
    
    print(f"\n\n✅ Indexed {len(rows)} items in {duration:.2f}s ({rate:.1f} items/s)")
    print(f"Database: {args.db}")

def handle_search(args):
    """Search the database."""
    if not os.path.exists(args.db):
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)

    embedder = OllamaEmbedder(DEFAULT_OLLAMA_HOST, args.model)
    if not embedder.check_connection():
        print(f"Error: Could not connect to Ollama at {DEFAULT_OLLAMA_HOST}")
        sys.exit(1)

    db = VectorDB(args.db)
    print(f"Loaded {len(db.vectors)} vectors from {args.db}")
    print(f"Query: '{args.query}' (Model: {args.model})")
    
    try:
        query_vec = embedder.get_embedding(args.query)
        results = db.search(query_vec, k=args.k)
        
        print("\nResults:")
        print("-" * 40)
        for id, score, meta in results:
            print(f"[{score:.4f}] {id}")
            # Try to print a meaningful summary
            display = meta.get("text", meta.get("Title", str(meta)))
            # Truncate if too long
            if len(str(display)) > 100:
                display = str(display)[:100] + "..."
            print(f"  {display}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Search failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Cicada Vector CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Index Command
    index_parser = subparsers.add_parser("index", help="Embed and index a file")
    index_parser.add_argument("file", help="Input file (.csv or .jsonl)")
    index_parser.add_argument("--db", default="vectors.jsonl", help="Output database path")
    index_parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    index_parser.set_defaults(func=handle_index)

    # Search Command
    search_parser = subparsers.add_parser("search", help="Search the database")
    search_parser.add_argument("query", help="Text to search for")
    search_parser.add_argument("--db", default="vectors.jsonl", help="Database path")
    search_parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model to use")
    search_parser.add_argument("-k", type=int, default=5, help="Number of results")
    search_parser.set_defaults(func=handle_search)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
