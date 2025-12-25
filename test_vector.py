import os
import random
import time
from cicada_vector import VectorDB

def test_db():
    db_path = "test_vectors.jsonl"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    db = VectorDB(db_path)
    
    # 1. Add vectors
    print("Adding 1000 vectors...")
    start = time.time()
    for i in range(1000):
        # Create random normalized vector
        vec = [random.random() for _ in range(128)]
        db.add(f"vec_{i}", vec, {"index": i})
    print(f"Added in {time.time() - start:.4f}s")
    
    # 2. Search
    print("Searching...")
    query = [random.random() for _ in range(128)]
    start = time.time()
    results = db.search(query, k=5)
    print(f"Search (Pure Python) in {time.time() - start:.4f}s")
    
    for id, score, meta in results:
        print(f"  {id}: {score:.4f} {meta}")
        
    # 3. Persistence check
    print("Reloading from disk...")
    db2 = VectorDB(db_path)
    assert len(db2.vectors) == 1000
    print("Persistence OK")
    
    os.remove(db_path)

if __name__ == "__main__":
    test_db()
