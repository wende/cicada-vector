"""
End-to-End Test Suite: Cicada Vector + Ollama
"""

import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from typing import List, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cicada_vector import VectorDB

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# We prefer a dedicated embedding model, but fallback to general purpose if needed
PREFERRED_MODELS = ["nomic-embed-text", "mxbai-embed-large", "llama3", "mistral"]

class OllamaClient:
    def __init__(self, host: str):
        self.host = host
        self.model: Optional[str] = None

    def check_connection(self) -> bool:
        """Check if Ollama is reachable."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags") as response:
                return response.status == 200
        except Exception:
            return False

    def find_best_model(self) -> Optional[str]:
        """Find the best available model for embeddings."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags") as response:
                data = json.loads(response.read().decode())
                available = [m["name"].split(":")[0] for m in data.get("models", [])]
                
                print(f"Found models: {available}")
                
                # Check preferred order
                for pref in PREFERRED_MODELS:
                    # Match exact or with :latest tag
                    matches = [m for m in available if m == pref or m.startswith(pref + ":")]
                    if matches:
                        return matches[0]
                        
                # Fallback to first available if list not empty
                if available:
                    return available[0]
                    
                return None
        except Exception as e:
            print(f"Error listing models: {e}")
            return None

    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single string."""
        url = f"{self.host}/api/embeddings"
        data = {
            "model": self.model,
            "prompt": text,
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            return result["embedding"]

def run_e2e_test():
    print(f"Testing connection to Ollama at {OLLAMA_HOST}...")
    client = OllamaClient(OLLAMA_HOST)
    
    if not client.check_connection():
        print("\n❌ FATAL: Could not connect to Ollama.")
        print(f"Please ensure Ollama is running at {OLLAMA_HOST}")
        print("Run: ollama serve")
        sys.exit(1)
        
    print("✓ Connection successful")
    
    print("Selecting model...")
    model = client.find_best_model()
    if not model:
        print("\n❌ FATAL: No models found in Ollama.")
        print(f"Please pull an embedding model:")
        print(f"Run: ollama pull {PREFERRED_MODELS[0]}")
        sys.exit(1)
        
    client.model = model
    print(f"✓ Using model: {model}")
    
    # Setup DB
    db_path = "e2e_vectors.jsonl"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = VectorDB(db_path)
    print(f"✓ Initialized VectorDB at {db_path}")
    
    # Test Data: Semantic Clusters
    documents = [
        # Coding / Python
        ("doc_1", "Python is a high-level programming language known for readability."),
        ("doc_2", "FastAPI and Flask are popular web frameworks for building APIs."),
        
        # Food / Cooking
        ("doc_3", "To make a perfect omelette, whisk eggs and cook on medium heat."),
        ("doc_4", "Fresh ingredients are essential for a delicious pasta sauce."),
        
        # Space / Science
        ("doc_5", "The Hubble telescope captures images of distant galaxies."),
        ("doc_6", "Mars rovers search for signs of ancient water on the red planet."),
    ]
    
    print(f"\nEmbedding and Indexing {len(documents)} documents...")
    start_time = time.time()
    
    for doc_id, text in documents:
        print(f"  Embedding: {doc_id}...", end="", flush=True)
        try:
            vector = client.get_embedding(text)
            db.add(doc_id, vector, {"text": text})
            print(" Done.")
        except Exception as e:
            print(f" Failed! {e}")
            sys.exit(1)
            
    print(f"✓ Indexing complete in {time.time() - start_time:.2f}s")
    
    # Test Cases: Query -> Expected Top Result
    test_cases = [
        ("coding in python", "doc_1"),
        ("cooking recipe", "doc_3"),
        ("astronomy and stars", "doc_5"),
    ]
    
    print("\nRunning Semantic Search Tests...")
    failures = 0
    
    for query, expected_id in test_cases:
        print(f"\nQuery: '{query}'")
        q_vec = client.get_embedding(query)
        results = db.search(q_vec, k=3)
        
        # Print top result
        top_id, score, meta = results[0]
        print(f"  Top Result: {top_id} (Score: {score:.4f})")
        print(f"  Text: {meta['text']}")
        
        if top_id == expected_id:
            print(f"  Result: ✅ PASS")
        else:
            print(f"  Result: ❌ FAIL (Expected {expected_id})")
            failures += 1
            
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
        
    if failures == 0:
        print("\n✨ All End-to-End Tests Passed! ✨")
        sys.exit(0)
    else:
        print(f"\n💀 {failures} Tests Failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_e2e_test()
