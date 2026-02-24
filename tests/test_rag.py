import os
import shutil
import unittest
from typing import List
from cicada_vector import VectorIndex


class MockEmbedder:
    max_chars = 900

    def embed(self, text: str) -> List[float]:
        v = [0.0] * 10
        if "login" in text: v[0] = 1.0
        if "auth" in text: v[1] = 1.0
        return v

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class TestVectorIndex(unittest.TestCase):
    def setUp(self):
        self.storage_dir = "test_rag_db"
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    def tearDown(self):
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    def test_keyword_search(self):
        db = VectorIndex(self.storage_dir, embedding_provider=MockEmbedder())
        content = "def login():\n    return True"
        db.add_file("auth.py", content)

        results = db.search("login", k=1)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['file'], "auth.py")
        self.assertTrue(results[0]['full_match'])  # keyword hit

    def test_precomputed_vector(self):
        db = VectorIndex(self.storage_dir, embedding_provider=MockEmbedder())
        content = "def login():\n    return True"
        vector = [1.0] * 10
        db.add_file("auth.py", content, vector=vector)

        results = db.search("login", k=1, query_vector=[1.0] * 10)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['file'], "auth.py")


if __name__ == "__main__":
    unittest.main()
