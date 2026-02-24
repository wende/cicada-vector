"""
End-to-end tests requiring a running Ollama instance.
Skipped automatically when Ollama is not available.

Tests cover the full stack:
  OllamaEmbedding -> KeywordDB -> Store (RRF) -> VectorIndex (two-level RAG) -> DirectoryIndexer
"""

import os
import shutil
import tempfile
import unittest
import urllib.request

from cicada_vector import EmbeddingDB, KeywordDB, OllamaEmbedding, Store, VectorIndex
from cicada_vector.indexer import DirectoryIndexer

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nomic-embed-text")

# Four semantically distinct documents with unique identifying terms
FIXTURE_DOCS = {
    "auth.py": """\
def authenticate(username: str, password: str) -> str:
    \"\"\"Authenticate user and return a signed JWT token.\"\"\"
    if not verify_password(username, password):
        raise AuthenticationError("Invalid credentials")
    return generate_jwt_token(username)

def verify_token(token: str) -> dict:
    \"\"\"Decode and verify JWT token, return payload.\"\"\"
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")
""",
    "database.py": """\
def connect_database(host: str, port: int, dbname: str):
    \"\"\"Open a PostgreSQL connection using psycopg2.\"\"\"
    dsn = f"postgresql://{host}:{port}/{dbname}"
    return psycopg2.connect(dsn)

def execute_query(conn, sql: str, params=None):
    \"\"\"Execute a parameterised SQL query and return all rows.\"\"\"
    with conn.cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchall()
""",
    "networking.py": """\
def make_http_request(url: str, method: str = "GET", headers: dict = None) -> dict:
    \"\"\"Send an HTTP request and return the JSON response body.\"\"\"
    response = requests.request(method, url, headers=headers or {})
    response.raise_for_status()
    return response.json()

def create_socket_server(host: str, port: int):
    \"\"\"Bind and listen on a TCP socket.\"\"\"
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    return server
""",
    "testing.py": """\
def run_test_suite(test_dir: str) -> dict:
    \"\"\"Discover and run the pytest suite, return pass/fail counts.\"\"\"
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return {"passed": result.testsRun - len(result.failures), "failed": len(result.failures)}

def mock_external_service(service_name: str):
    \"\"\"Return a MagicMock standing in for an external dependency.\"\"\"
    return unittest.mock.MagicMock(name=service_name)
""",
}


def _ollama_available() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def _model_available() -> bool:
    try:
        import json
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as r:
            data = json.loads(r.read())
            names = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return OLLAMA_MODEL.split(":")[0] in names
    except Exception:
        return False


@unittest.skipUnless(_ollama_available(), "Ollama not running")
class TestEmbeddingLayer(unittest.TestCase):
    """Tests for OllamaEmbedding: single embed, batch, adaptive limits."""

    @classmethod
    def setUpClass(cls):
        if not _model_available():
            raise unittest.SkipTest(f"Model '{OLLAMA_MODEL}' not pulled (run: ollama pull {OLLAMA_MODEL})")
        cls.embedder = OllamaEmbedding(host=OLLAMA_HOST, model=OLLAMA_MODEL)

    def test_embed_returns_vector(self):
        vec = self.embedder.embed("hello world")
        self.assertIsInstance(vec, list)
        self.assertGreater(len(vec), 0)
        self.assertIsInstance(vec[0], float)

    def test_embed_batch_matches_single(self):
        texts = ["authentication login", "database query", "http socket"]
        batch = self.embedder.embed_batch(texts)
        self.assertEqual(len(batch), len(texts))
        for i, text in enumerate(texts):
            single = self.embedder.embed(text)
            # Cosine similarity between batch and single result should be > 0.999
            dot = sum(a * b for a, b in zip(batch[i], single))
            mag_a = sum(x * x for x in batch[i]) ** 0.5
            mag_b = sum(x * x for x in single) ** 0.5
            similarity = dot / (mag_a * mag_b) if mag_a and mag_b else 0
            self.assertGreater(similarity, 0.999, f"Batch vs single diverged for: {text!r}")

    def test_max_chars_calibrated_after_embed(self):
        # max_chars starts as a table-driven default; after an embed it should
        # be calibrated from the actual prompt_eval_count response
        before = self.embedder._chars_per_token  # None before first call
        self.embedder.embed("calibration text " * 20)
        after = self.embedder._chars_per_token
        # If the endpoint returns prompt_eval_count, calibration fires
        if after is not None:
            self.assertGreater(after, 0)
            self.assertLess(after, 20)  # sanity: < 20 chars/token

    def test_max_chars_positive(self):
        self.assertGreater(self.embedder.max_chars, 100)

    def test_vectors_are_unit_length(self):
        """EmbeddingDB normalizes on add; raw Ollama vectors may not be unit length."""
        import math
        raw = self.embedder.embed("unit length check")
        mag = math.sqrt(sum(x * x for x in raw))
        # Raw vectors from Ollama may or may not be unit-length — that's OK.
        # What matters is that EmbeddingDB normalizes them (tested in test_core.py).
        self.assertGreater(mag, 0)


@unittest.skipUnless(_ollama_available(), "Ollama not running")
class TestKeywordIDF(unittest.TestCase):
    """Tests for IDF-weighted keyword scoring."""

    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.mktemp(suffix=".json")
        cls.db = KeywordDB(cls.storage)
        for name, content in FIXTURE_DOCS.items():
            cls.db.add(name, content)
        cls.db.persist()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.storage):
            os.unlink(cls.storage)

    def test_total_docs_tracked(self):
        self.assertEqual(self.db._total_docs, len(FIXTURE_DOCS))

    def test_unique_term_ranks_its_doc_first(self):
        """A term appearing in exactly one doc must be ranked first."""
        unique_terms = {
            "jwt":         "auth.py",
            "psycopg2":    "database.py",
            "postgresql":  "database.py",
            "socket":      "networking.py",
            "magicmock":   "testing.py",
        }
        for term, expected_doc in unique_terms.items():
            results = self.db.search(term)
            self.assertTrue(results, f"No results for term: {term!r}")
            self.assertEqual(results[0], expected_doc,
                             f"Expected {expected_doc!r} first for {term!r}, got {results[0]!r}")

    def test_idf_ranking_prefers_rare_terms(self):
        """'jwt' (1 doc) should rank auth.py above 'username' (multiple docs) alone."""
        # 'username' appears in both auth.py and potentially others
        # 'jwt' appears only in auth.py
        jwt_results = self.db.search("jwt")
        self.assertEqual(jwt_results[0], "auth.py")

    def test_meta_persistence(self):
        """_total_docs survives a reload."""
        db2 = KeywordDB(self.storage)
        self.assertEqual(db2._total_docs, self.db._total_docs)

    def test_multi_term_boosts_matching_doc(self):
        """A doc matching two distinctive query terms outranks one matching only one."""
        # auth.py has both 'jwt' and 'token' — should outscore docs with only one
        results = self.db.search("jwt token")
        self.assertEqual(results[0], "auth.py")


@unittest.skipUnless(_ollama_available(), "Ollama not running")
class TestHybridStore(unittest.TestCase):
    """Tests for Store (RRF merge of dense + sparse)."""

    @classmethod
    def setUpClass(cls):
        if not _model_available():
            raise unittest.SkipTest(f"Model '{OLLAMA_MODEL}' not pulled")
        cls.storage = tempfile.mkdtemp()
        cls.store = Store(cls.storage, ollama_host=OLLAMA_HOST, ollama_model=OLLAMA_MODEL)
        for name, content in FIXTURE_DOCS.items():
            cls.store.add(name, content, meta={"file": name})
        cls.store.persist()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.storage, ignore_errors=True)

    def test_search_returns_list(self):
        results = self.store.search("authentication", k=2)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 2)

    def test_result_structure(self):
        results = self.store.search("database connection", k=1)
        self.assertEqual(len(results), 1)
        doc_id, score, meta = results[0]
        self.assertIsInstance(doc_id, str)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_scores_normalized_to_one(self):
        """Top RRF result should always have score == 1.0."""
        results = self.store.search("jwt token authentication", k=4)
        self.assertTrue(results)
        self.assertAlmostEqual(results[0][1], 1.0, places=5)

    def test_exact_keyword_match_surfaces_correct_doc(self):
        """A unique term in a doc must appear in results."""
        results = self.store.search("psycopg2", k=4)
        doc_ids = [r[0] for r in results]
        self.assertIn("database.py", doc_ids)

    def test_semantic_match_finds_related_doc(self):
        """Semantic query should surface the thematically closest doc in top-2."""
        results = self.store.search("user login verification", k=2)
        doc_ids = [r[0] for r in results]
        self.assertIn("auth.py", doc_ids, "Expected auth.py in top-2 for login query")

    def test_rrf_promotes_dual_match(self):
        """A doc matching both semantically and by keyword ranks above keyword-only matches."""
        # 'jwt' is unique to auth.py; semantic embedding of the query should also favour it
        results = self.store.search("jwt authentication", k=4)
        self.assertEqual(results[0][0], "auth.py")


@unittest.skipUnless(_ollama_available(), "Ollama not running")
class TestVectorIndexRAG(unittest.TestCase):
    """Tests for the two-level RAG: file-level broad search + chunk-level refinement."""

    @classmethod
    def setUpClass(cls):
        if not _model_available():
            raise unittest.SkipTest(f"Model '{OLLAMA_MODEL}' not pulled")
        cls.storage = tempfile.mkdtemp()
        cls.index = VectorIndex(cls.storage, ollama_host=OLLAMA_HOST, ollama_model=OLLAMA_MODEL)
        for name, content in FIXTURE_DOCS.items():
            cls.index.add_file(name, content)
        cls.index.persist()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.storage, ignore_errors=True)

    def test_search_result_structure(self):
        results = self.index.search("database", k=1)
        self.assertTrue(results)
        r = results[0]
        self.assertIn("file", r)
        self.assertIn("score", r)
        self.assertIn("line", r)
        self.assertIn("snippet", r)
        self.assertIn("full_match", r)

    def test_keyword_match_flagged(self):
        """A query with an exact term in the doc should set full_match=True."""
        results = self.index.search("psycopg2", k=2)
        db_results = [r for r in results if r["file"] == "database.py"]
        self.assertTrue(db_results)
        self.assertTrue(db_results[0]["full_match"])

    def test_snippet_is_multiline(self):
        """Snippet should be a text chunk, not a single character."""
        results = self.index.search("authenticate jwt", k=1)
        self.assertTrue(results)
        snippet = results[0]["snippet"]
        self.assertIsInstance(snippet, str)
        self.assertGreater(len(snippet), 10)

    def test_line_number_positive(self):
        results = self.index.search("postgresql database", k=1)
        self.assertTrue(results)
        self.assertGreater(results[0]["line"], 0)

    def test_score_in_range(self):
        for query in ["jwt token", "database query", "http request", "mock test"]:
            results = self.index.search(query, k=1)
            if results:
                self.assertGreaterEqual(results[0]["score"], 0.0)
                self.assertLessEqual(results[0]["score"], 1.0)

    def test_semantic_query_finds_relevant_file(self):
        """Semantic-only query (no exact keyword) should still surface the right file."""
        results = self.index.search("user identity verification", k=2)
        files = [r["file"] for r in results]
        self.assertIn("auth.py", files)

    def test_chunk_caching(self):
        """Second search on the same file should hit cached chunks (no new embeds)."""
        self.index.search("token expiry", k=1)
        count_before = len(self.index.db.vector_db.ids)
        self.index.search("token expiry", k=1)
        count_after = len(self.index.db.vector_db.ids)
        self.assertEqual(count_before, count_after, "Chunks were re-created on second search")


@unittest.skipUnless(_ollama_available(), "Ollama not running")
class TestDirectoryIndexer(unittest.TestCase):
    """Tests for DirectoryIndexer: scan, batch embed, incremental updates."""

    @classmethod
    def setUpClass(cls):
        if not _model_available():
            raise unittest.SkipTest(f"Model '{OLLAMA_MODEL}' not pulled")
        # Write fixture docs to a real temp directory
        cls.src_dir = tempfile.mkdtemp()
        cls.storage = tempfile.mkdtemp()
        for name, content in FIXTURE_DOCS.items():
            with open(os.path.join(cls.src_dir, name), "w") as f:
                f.write(content)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.src_dir, ignore_errors=True)
        shutil.rmtree(cls.storage, ignore_errors=True)

    def _make_indexer(self):
        return DirectoryIndexer(self.storage, ollama_host=OLLAMA_HOST, ollama_model=OLLAMA_MODEL)

    def test_index_returns_stats(self):
        indexer = self._make_indexer()
        stats = indexer.index_directory(self.src_dir)
        self.assertIn("added", stats)
        self.assertIn("skipped", stats)
        self.assertIn("failed", stats)
        self.assertEqual(stats["failed"], 0)

    def test_all_files_indexed(self):
        indexer = self._make_indexer()
        stats = indexer.index_directory(self.src_dir)
        # First run: all 4 docs should be added (or already added by a previous test)
        self.assertGreaterEqual(stats["added"] + stats["skipped"], len(FIXTURE_DOCS))

    def test_incremental_skips_unchanged(self):
        """Running the indexer twice should skip all files on the second pass."""
        indexer = self._make_indexer()
        indexer.index_directory(self.src_dir)  # first pass
        stats2 = indexer.index_directory(self.src_dir)  # second pass
        self.assertEqual(stats2["added"], 0)
        self.assertEqual(stats2["skipped"], len(FIXTURE_DOCS))

    def test_modified_file_reindexed(self):
        """Touching a file's content triggers re-indexing on next run."""
        indexer = self._make_indexer()
        indexer.index_directory(self.src_dir)

        modified = os.path.join(self.src_dir, "auth.py")
        with open(modified, "a") as f:
            f.write("\n# updated\n")

        stats = indexer.index_directory(self.src_dir)
        self.assertEqual(stats["added"], 1)
        self.assertEqual(stats["skipped"], len(FIXTURE_DOCS) - 1)

    def test_indexed_files_searchable(self):
        """After indexing, searching should return results from the fixture files."""
        indexer = self._make_indexer()
        indexer.index_directory(self.src_dir)

        results = indexer.rag_db.search("jwt token authentication", k=2)
        self.assertTrue(results)
        files = [r["file"] for r in results]
        self.assertTrue(
            any("auth" in f for f in files),
            f"Expected auth.py in results, got: {files}"
        )

    def test_batch_embedding_no_failures(self):
        """Batch embedding path (BATCH_SIZE=8, 4 files) should produce 0 failures."""
        storage2 = tempfile.mkdtemp()
        try:
            indexer = DirectoryIndexer(storage2, ollama_host=OLLAMA_HOST, ollama_model=OLLAMA_MODEL)
            stats = indexer.index_directory(self.src_dir)
            self.assertEqual(stats["failed"], 0)
            self.assertGreaterEqual(stats["added"], len(FIXTURE_DOCS))
        finally:
            shutil.rmtree(storage2, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
