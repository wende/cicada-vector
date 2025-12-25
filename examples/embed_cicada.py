"""
Batch embed functions from a real Cicada index.json
"""

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

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "nomic-embed-text"
INDEX_PATH = "tests/fixtures/real_project_index/index.json"
VECTORS_PATH = "cicada_vectors.jsonl"
LIMIT = 500 # Limit for test run

class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host = host
        self.model = model

    def get_embedding(self, text: str) -> List[float]:
        url = f"{self.host}/api/embeddings"
        data = {"model": self.model, "prompt": text}
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

    print(f"Loading index from {INDEX_PATH}...")
    with open(INDEX_PATH, 'r') as f:
        index = json.load(f)

    db = VectorDB(VECTORS_PATH)
    embedder = OllamaEmbedder(OLLAMA_HOST, OLLAMA_MODEL)

    functions_to_embed = []
    
    # 1. Collect functions
    for mod_name, mod_data in index.get("modules", {}).items():
        file_path = mod_data.get("file", "unknown")
        for func in mod_data.get("functions", []):
            func_name = func.get("name")
            doc = func.get("doc", "")
            keywords = list(func.get("keywords", {}).keys())
            
            # Create a meaningful description for embedding
            text = f"Function: {func_name} in module {mod_name}. File: {file_path}. Description: {doc}. Keywords: {', '.join(keywords)}"
            
            functions_to_embed.append({
                "id": f"{mod_name}.{func_name}",
                "text": text,
                "meta": {
                    "module": mod_name,
                    "function": func_name,
                    "file": file_path,
                    "line": func.get("line")
                }
            })
            
            if len(functions_to_embed) >= LIMIT:
                break
        if len(functions_to_embed) >= LIMIT:
            break

    print(f"Found {len(functions_to_embed)} functions to embed.")
    
    # 2. Embed and Store
    start_time = time.time()
    for i, item in enumerate(functions_to_embed):
        print(f"\r[{i+1}/{len(functions_to_embed)}] Embedding {item['id']}...", end="", flush=True)
        try:
            vector = embedder.get_embedding(item['text'])
            db.add(item['id'], vector, item['meta'])
        except Exception as e:
            print(f"\nError embedding {item['id']}: {e}")
            continue

    print(f"\n✓ Finished embedding {len(functions_to_embed)} functions in {time.time() - start_time:.2f}s")
    print(f"Vectors stored in {VECTORS_PATH}")

if __name__ == "__main__":
    main()
