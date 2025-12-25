import os
import shutil
import unittest
from typing import List
from cicada_vector import HybridDB

class TestHybridDB(unittest.TestCase):
    def setUp(self):
        self.storage_dir = "test_hybrid_db"
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
            
    def tearDown(self):
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    def _mock_embedding(self, text: str) -> List[float]:
        vec = [0.0] * 10
        if "python" in text.lower(): vec[0] = 1.0
        return vec

    def test_hybrid_search(self):
        db = HybridDB(self.storage_dir)
        db.add("doc_1", self._mock_embedding("python"), "python", {"text": "python"})
        
        results = db.search("python", self._mock_embedding("python"), k=1)
        self.assertEqual(results[0][0], "doc_1")

if __name__ == "__main__":
    unittest.main()
