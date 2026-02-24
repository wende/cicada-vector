"""
Embedding provider abstraction.
Handles getting embeddings from various sources (Ollama, OpenAI, etc.)
"""

import json
import urllib.request
from typing import List, Optional, Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts in a single call."""
        ...


class OllamaEmbedding:
    """Ollama embedding provider using the /api/embed endpoint."""

    # Known context sizes (in tokens) for common embedding models.
    # Used to compute max_chars before adaptive calibration kicks in.
    _MODEL_MAX_TOKENS: dict = {
        "nomic-embed-text": 2048,
        "nomic-embed-text-v1.5": 8192,
        "mxbai-embed-large": 512,
        "all-minilm": 256,
        "bge-m3": 8192,
        "snowflake-arctic-embed": 512,
    }
    _DEFAULT_MAX_TOKENS = 512

    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.host = host
        self.model = model
        # Calibrated chars/token ratio, updated after first successful embed
        self._chars_per_token: Optional[float] = None
        # Strip version tag (e.g. "nomic-embed-text:v1.5" -> "nomic-embed-text")
        base_model = model.split(":")[0]
        self._max_tokens: int = self._MODEL_MAX_TOKENS.get(base_model, self._DEFAULT_MAX_TOKENS)

    @property
    def max_chars(self) -> int:
        """Maximum characters to embed, based on token limit and calibrated chars/token ratio."""
        ratio = self._chars_per_token or 4.0  # 4 chars/token is a conservative default for code
        return int(self._max_tokens * ratio * 0.9)  # 10% safety margin

    def _calibrate(self, texts: List[str], prompt_eval_count: int) -> None:
        """Update chars/token ratio from a successful embed response."""
        total_chars = sum(len(t) for t in texts)
        if prompt_eval_count > 0 and total_chars > 100:
            measured = total_chars / prompt_eval_count
            if self._chars_per_token is None:
                self._chars_per_token = measured
            else:
                # Exponential moving average to smooth out variation
                self._chars_per_token = 0.8 * self._chars_per_token + 0.2 * measured

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts in a single API call."""
        url = f"{self.host}/api/embed"
        data = {"model": self.model, "input": texts}
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                embeddings = result["embeddings"]
                if result.get("prompt_eval_count"):
                    self._calibrate(texts, result["prompt_eval_count"])
                return embeddings
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.embed_batch([text])[0]
