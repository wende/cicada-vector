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
        
        content = """
def login(username, password):
    # Validate user credentials
    if not user.exists():
        return False
    return True
"""
        # Mock vector (all zeros as we rely on text matching for this test logic mostly)
        vec = [0.0] * 10
        db.add_file("auth.py", content, vec)
        
        # Search for "user credentials"
        # We simulate a vector match by manually adding it to the underlying DB for this test?
        # Or better, we mock the search method of the underlying HybridDB?
        # For simplicity, let's just rely on Keyword match part of HybridDB finding "user"
        
        q = "user credentials"
        # query_terms in RagDB.search uses re.findall which will find "user", "credentials"
        
        results = db.search(q, vec, k=1)
        
        self.assertTrue(len(results) > 0)
        res = results[0]
        self.assertEqual(res['file'], "auth.py")
        self.assertIn("def login", res['snippet'])
        # Line number should be where density is highest (line 1, 2 or 3)
        self.assertIn(res['line'], [1, 2, 3])

if __name__ == "__main__":
    unittest.main()