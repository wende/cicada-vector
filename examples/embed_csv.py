"""
Benchmark: Embed a CSV file using Ollama + Cicada Vector
"""

import csv
import json
import os
import sys
import time
import urllib.request
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cicada_vector import VectorDB

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "nomic-embed-text"
CSV_PATH = "cicada-vector/Activities.csv"
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
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return

    # Clean up old DB
    if os.path.exists(VECTORS_PATH):
        os.remove(VECTORS_PATH)

    db = VectorDB(VECTORS_PATH)
    
    print(f"Reading {CSV_PATH}...")
    
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Create a rich semantic string
            text = f"{row['Type']} of {row['Title']} in {row['Application']} ({row['Top-Level Project']})"
            rows.append({
                "id": f"row_{i}",
                "text": text,
                "meta": row
            })

    total_rows = len(rows)
    print(f"Embedding {total_rows} rows using {OLLAMA_MODEL}...")
    
    start_time = time.time()
    
    # Process in sequence (Ollama handles one request at a time usually unless batched)
    for i, item in enumerate(rows):
        # Progress bar
        if i % 10 == 0:
            print(f"\r[{i}/{total_rows}] {(i/total_rows)*100:.1f}%", end="", flush=True)
            
        try:
            vector = get_embedding(item['text'])
            db.add(item['id'], vector, item['meta'])
        except Exception as e:
            print(f"\nError embedding row {i}: {e}")

    duration = time.time() - start_time
    rate = total_rows / duration
    
    print(f"\n\n✅ Done!")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Rate: {rate:.2f} items/second")
    print(f"Database saved to {VECTORS_PATH}")

if __name__ == "__main__":
    main()
