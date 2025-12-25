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
        # Simple deterministic embedding based on word presence
        # "python" -> dim 0, "react" -> dim 1
        vec = [0.0] * 10
        text = text.lower()
        if "python" in text: vec[0] = 1.0
        if "react" in text: vec[1] = 1.0
        if "data" in text: vec[2] = 1.0
        return vec

    def test_hybrid_search(self):
        db = HybridDB(self.storage_dir)
        
        documents = [
            ("doc_1", "Python is great for data science.", "python data science"),
            ("doc_2", "React is a javascript library for UI.", "react javascript ui"),
            ("doc_3", "Project Thenvoi is a new initiative.", "project thenvoi initiative"),
        ]
        
        for doc_id, text, keywords in documents:
            vec = self._mock_embedding(text)
            # text field used for keyword indexing
            db.add(doc_id, vec, text, {"text": text})
            
        # Test 1: Vector Match ("python")
        q1 = "python"
        q1_vec = self._mock_embedding(q1)
        results = db.search(q1, q1_vec, k=1)
        self.assertEqual(results[0][0], "doc_1")
        
        # Test 2: Keyword Match ("Thenvoi")
        # Vector will be empty/zeros (no semantic match in our mock)
        q2 = "Thenvoi"
        q2_vec = self._mock_embedding(q2) 
        results = db.search(q2, q2_vec, k=1)
        
        # Should match because of keyword db
        self.assertEqual(results[0][0], "doc_3")
        # Score should be boosted (at least 0.5 base for keyword only)
        self.assertGreaterEqual(results[0][1], 0.5)

if __name__ == "__main__":
    unittest.main()