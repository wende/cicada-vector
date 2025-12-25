import os
import shutil
import unittest
import random
from cicada_vector import VectorDB

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_vectors.jsonl"
        # Ensure clean state
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_add_and_search(self):
        db = VectorDB(self.db_path)
        
        # 1. Add vectors
        for i in range(100):
            # Create distinct vectors using one-hot-ish encoding
            # vec_i has 1.0 at index i
            vec = [0.0] * 100
            vec[i] = 1.0
            db.add(f"vec_{i}", vec, {"index": i})
            
        # 2. Search
        # Search for vector identical to vec_50
        query = [0.0] * 100
        query[50] = 1.0
        
        results = db.search(query, k=5)
        
        # Top result should be vec_50 (score 1.0)
        self.assertEqual(results[0][0], "vec_50")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)
        
    def test_persistence(self):
        db = VectorDB(self.db_path)
        vec = [1.0] * 128
        db.add("test_vec", vec)
        
        # Reload
        db2 = VectorDB(self.db_path)
        self.assertEqual(len(db2.vectors), 1)
        self.assertEqual(db2.ids[0], "test_vec")

if __name__ == "__main__":
    unittest.main()