import os
import shutil
import unittest
from cicada_vector import RagDB

class TestRagDB(unittest.TestCase):
    def setUp(self):
        self.storage_dir = "test_rag_db"
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
            
    def tearDown(self):
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)

    def test_search_and_scan(self):
        db = RagDB(self.storage_dir)
        content = "def login():\n    return True"
        db.add_file("auth.py", content, [1.0]*10)
        
        results = db.search("login", [1.0]*10, k=1)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]['file'], "auth.py")

if __name__ == "__main__":
    unittest.main()
