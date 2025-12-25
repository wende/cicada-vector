"""
Test Poor Man's RAG: File Search -> Line Scan
"""

import os
import shutil
import sys
import json
import urllib.request
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from cicada_vector import RagDB

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "nomic-embed-text"
STORAGE_DIR = "test_rag_db"

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
        
    db = RagDB(STORAGE_DIR)
    
    # Dataset: Fake code files
    files = [
        ("auth.py", """
def login(username, password):
    # Validate user credentials
    if not user.exists():
        return False
    return True

def logout():
    session.clear()
"""),
        ("database.py", """
class Database:
    def connect(self):
        print("Connecting...")
        
    def query(self, sql):
        # Execute raw SQL
        return []
"""),
        ("readme.md", """
# Project Docs

## Installation
Run `pip install -r requirements.txt`

## Authentication
Use the `auth.py` module to handle login.
""")
    ]
    
    print(f"Indexing {len(files)} files...")
    
    for filename, content in files:
        # Create a "Representation" for embedding: 
        # Filename + first few lines + docstrings is usually a good proxy
        representation = f"{filename}\n{content}" 
        vec = get_embedding(representation)
        db.add_file(filename, content, vec)
        
    print("\n--- Search 1: 'database connection' ---")
    q1 = "database connection"
    q1_vec = get_embedding(q1)
    results = db.search(q1, q1_vec, k=1)
    
    for res in results:
        print(f"File: {res['file']} (Line {res['line']})")
        print(f"Snippet:\n{res['snippet']}")
        
        if "class Database" in res['snippet'] or "def connect" in res['snippet']:
             print("✅ Success: Found DB connection code")
        else:
             print("❌ Fail: Missed the connection logic")

    print("\n--- Search 2: 'user login' ---")
    q2 = "user login"
    q2_vec = get_embedding(q2)
    results = db.search(q2, q2_vec, k=1)
    
    for res in results:
        print(f"File: {res['file']} (Line {res['line']})")
        print(f"Snippet:\n{res['snippet']}")
        
        if "def login" in res['snippet']:
            print("✅ Success: Found login function")
        else:
            print("❌ Fail: Missed login function")

if __name__ == "__main__":
    main()
