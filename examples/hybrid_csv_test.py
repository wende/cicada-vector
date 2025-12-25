"""
Benchmark: Hybrid Search on Activities.csv
"""

import csv
import json
import os
import sys
import time
import urllib.request
import shutil
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cicada_vector import HybridDB

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "nomic-embed-text"
CSV_PATH = "cicada-vector/Activities.csv"
STORAGE_DIR = "hybrid_csv_db"

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
    if os.path.exists(STORAGE_DIR):
        shutil.rmtree(STORAGE_DIR)

    db = HybridDB(STORAGE_DIR)
    
    print(f"Reading {CSV_PATH}...")
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Rich text for embedding
            text = f"{row['Type']} of {row['Title']} in {row['Application']} ({row['Top-Level Project']})"
            # Searchable text for keywords (concatenate fields including Path)
            searchable_text = f"{row['Title']} {row['Application']} {row['Type']} {row['Top-Level Project']} {row['Path']}"
            
            rows.append({
                "id": f"row_{i}",
                "text": text,
                "searchable": searchable_text,
                "meta": row
            })

    total_rows = len(rows)
    print(f"Indexing {total_rows} rows (Hybrid)...")
    
    start_time = time.time()
    for i, item in enumerate(rows):
        if i % 50 == 0:
            print(f"\r[{i}/{total_rows}]", end="", flush=True)
        try:
            vector = get_embedding(item['text'])
            db.add(item['id'], vector, item['searchable'], item['meta'])
        except Exception:
            continue

    print(f"\nDone in {time.time() - start_time:.2f}s")
    
    # TEST: "Thenvoi"
    query = "thenvoi"
    print(f"\nQuerying Hybrid DB for: '{query}'...")
    
    q_vec = get_embedding(query)
    results = db.search(query, q_vec, k=5)
    
    for id, score, meta in results:
        print(f"  [{score:.4f}] {meta['Title']} ({meta['Application']})")

if __name__ == "__main__":
    main()
