"""
Embedding provider abstraction.
Handles getting embeddings from various sources (Ollama, OpenAI, etc.)
"""

import json
import urllib.request
from typing import List, Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> List[float]:
        """Get embedding for text."""
        ...


class OllamaEmbedding:
    """Ollama embedding provider."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.host = host
        self.model = model

    def embed(self, text: str) -> List[float]:
        """Get embedding from Ollama."""
        url = f"{self.host}/api/embeddings"
        data = {"model": self.model, "prompt": text}
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                return result["embedding"]
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")
