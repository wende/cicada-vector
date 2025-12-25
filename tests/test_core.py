import os
import unittest
from cicada_vector import VectorDB

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_vectors.jsonl"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_add_and_search(self):
        db = VectorDB(self.db_path)
        for i in range(10):
            vec = [0.0] * 10
            vec[i] = 1.0
            db.add(f"vec_{i}", vec, {"index": i})
            
        query = [0.0] * 10
        query[5] = 1.0
        results = db.search(query, k=1)
        self.assertEqual(results[0][0], "vec_5")
        
    def test_persistence(self):
        db = VectorDB(self.db_path)
        db.add("test", [1.0]*10)
        db2 = VectorDB(self.db_path)
        self.assertEqual(len(db2.vectors), 1)

if __name__ == "__main__":
    unittest.main()
