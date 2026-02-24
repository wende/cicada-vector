import os
import unittest
from cicada_vector import EmbeddingDB

class TestEmbeddingDB(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_vectors.jsonl"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_add_and_search(self):
        db = EmbeddingDB(self.db_path)
        for i in range(10):
            vec = [0.0] * 10
            vec[i] = 1.0
            db.add(f"vec_{i}", vec, {"index": i})
            
        query = [0.0] * 10
        query[5] = 1.0
        results = db.search(query, k=1)
        self.assertEqual(results[0][0], "vec_5")
        
    def test_persistence(self):
        db = EmbeddingDB(self.db_path)
        db.add("test", [1.0]*10)
        db2 = EmbeddingDB(self.db_path)
        self.assertEqual(len(db2.vectors), 1)

    def test_meta_always_dict_on_add(self):
        """meta must be a dict regardless of what is passed."""
        db = EmbeddingDB(self.db_path)
        db.add("str_meta", [1.0] * 10, "i am a string")  # type: ignore
        db.add("none_meta", [0.5] * 10, None)
        db.add("dict_meta", [0.1] * 10, {"key": "value"})

        for m in db.metadata:
            self.assertIsInstance(m, dict)

    def test_meta_always_dict_after_reload(self):
        """Non-dict meta stored on disk (legacy data) is coerced to dict on load."""
        import json

        # Write a record with string meta directly to mimic old/corrupted data
        with open(self.db_path, 'w') as f:
            f.write(json.dumps({"id": "old", "vector": [1.0] * 10, "meta": "raw string"}) + '\n')
            f.write(json.dumps({"id": "new", "vector": [0.5] * 10, "meta": {"key": "val"}}) + '\n')
            f.write(json.dumps({"id": "null", "vector": [0.1] * 10, "meta": None}) + '\n')

        db = EmbeddingDB(self.db_path)
        self.assertEqual(len(db.metadata), 3)
        for m in db.metadata:
            self.assertIsInstance(m, dict)

if __name__ == "__main__":
    unittest.main()
