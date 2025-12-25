"""
Search for 'thenvoi' in indexed Activities CSV
"""

import json
import os
import sys
import urllib.request
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cicada_vector import VectorDB

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "nomic-embed-text"
VECTORS_PATH = "csv_vectors.jsonl"

def get_embedding(text: str) -> List[float]:
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

def main():
    if not os.path.exists(VECTORS_PATH):
        print(f"Error: Vectors not found.")
        return

    db = VectorDB(VECTORS_PATH)
    
    q = "thenvoi"
    print(f"Querying for: '{q}'...")
    
    q_vec = get_embedding(q)
    results = db.search(q_vec, k=5)
    
    for id, score, meta in results:
        print(f"  [{score:.4f}] {meta['Title']} ({meta['Application']})")
        print(f"    Project: {meta['Top-Level Project']}, Type: {meta['Type']}")

if __name__ == "__main__":
    main()
