"""
Batch embed functions from a real Cicada index.json
"""

import json
import os
import sys
import time
import urllib.request
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from cicada_vector import VectorDB

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "nomic-embed-text"
INDEX_PATH = "index.json"
VECTORS_PATH = "cicada_vectors.jsonl"

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
    if not os.path.exists(INDEX_PATH):
        print(f"Error: Index not found at {INDEX_PATH}")
        return

    with open(INDEX_PATH, 'r') as f:
        index = json.load(f)

    db = VectorDB(VECTORS_PATH)
    
    # ... logic here ...
    print("Example script ready. Pass a real index.json to use.")

if __name__ == "__main__":
    main()