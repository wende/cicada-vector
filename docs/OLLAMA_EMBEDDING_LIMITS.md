# Ollama Embedding API Limits - Research Findings

**Date:** December 2025
**Models Tested:** nomic-embed-text, all-minilm
**Ollama Version:** 0.13.1

## Executive Summary

Ollama's embedding API has **model-specific hard limits** that are **not configurable** and **not documented** in model metadata. These limits vary significantly between models and use different failure modes (explicit errors vs silent truncation).

### Key Findings

| Model | Token Limit | Character Limit | Failure Mode | Detectable |
|-------|------------|-----------------|--------------|------------|
| nomic-embed-text | 510 tokens | ~2,550 chars | HTTP 500 EOF error | Yes (error thrown) |
| all-minilm | 254 tokens | ~3,000 chars | Silent truncation | Yes (via prompt_eval_count) |

**Critical Discovery:** The **new `/api/embed` endpoint** returns `prompt_eval_count` which tells us exactly how many tokens were processed, enabling detection of both error cases and silent truncation.

---

## Problem Statement

Our codebase had hardcoded `MAX_CHARS = 900` throughout (`rag.py`, `git_indexer.py`) to avoid Ollama 500 errors. This was:

1. **Too conservative** - We were truncating at 900 chars when nomic-embed-text accepts 2,550 chars
2. **Not model-agnostic** - Different models have wildly different limits
3. **Brittle** - Would break if models or Ollama versions change
4. **Undocumented** - No explanation of why 900 was chosen

We needed to understand **why** the limit exists and find a **robust, adaptive solution**.

---

## Investigation Process

### 1. Initial Hypothesis: `num_ctx` Controls Limit

**Tested:** Setting `num_ctx` parameter in Modelfile or API `options`

```python
# Test with different num_ctx values
data = {
    "model": "nomic-embed-text",
    "prompt": long_text,
    "options": {"num_ctx": 8192}
}
```

**Result:** No effect. Models fail at same character count regardless of `num_ctx`.

**Findings:**
- nomic-embed-text has `num_ctx=8192` but fails at ~2,550 chars
- all-minilm has `num_ctx=256` but accepts 10,000+ chars
- **Conclusion:** `num_ctx` does NOT control embedding input length

---

### 2. Character vs Token Limits

**Tested:** Sending repeated "word " pattern vs varied text

**Results:**
```
nomic-embed-text:
  - "word " * 500 = 2,500 chars -> Works (500 tokens)
  - "word " * 510 = 2,550 chars -> Works (510 tokens)
  - "word " * 511 = 2,555 chars -> Fails (EOF error)

Calculation: 2,550 chars / 510 tokens = 5.0 chars/token for this pattern
```

**Conclusion:** Limit is **token-based**, not character-based. Approximately **510 tokens** for nomic-embed-text.

---

### 3. Error Analysis

**nomic-embed-text at 2,600 chars:**
```json
{
  "error": "do embedding request: Post \"http://127.0.0.1:XXXXX/embedding\": EOF"
}
```

- HTTP 500 Internal Server Error
- Generic "EOF" message (connection closed unexpectedly)
- No useful metadata in Ollama server logs
- Occurs in llama.cpp backend, not Ollama API layer

**all-minilm at 5,000 chars:**
- No error thrown
- Returns HTTP 200 with embedding
- **Silent truncation** - only processes first ~254 tokens
- No indication in response that truncation occurred (with old API)

---

### 4. API Endpoint Comparison

#### Old Endpoint: `/api/embeddings` (Legacy)

**Request:**
```json
{
  "model": "nomic-embed-text",
  "prompt": "text here"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, ...]
}
```

**Limitations:**
- No token count
- No truncation metadata
- `truncate` parameter has no effect on nomic-embed-text
- Cannot detect silent truncation

#### New Endpoint: `/api/embed` (Recommended)

**Request:**
```json
{
  "model": "nomic-embed-text",
  "input": "text here",
  "truncate": false
}
```

**Response:**
```json
{
  "model": "nomic-embed-text",
  "embeddings": [[0.123, -0.456, ...]],
  "total_duration": 467000000,
  "load_duration": 123000000,
  "prompt_eval_count": 400
}
```

**Benefits:**
- **`prompt_eval_count`** - Actual tokens processed!
- **`truncate=false`** - Returns error instead of silent truncation (for all-minilm)
- Can detect if fewer tokens processed than expected
- Can calculate actual chars/token ratio

---

### 5. `truncate` Parameter Behavior

**With `truncate=false` on all-minilm:**
```
Input: 5,000 chars (exceeds 254 token limit)

Response: HTTP 400
{
  "error": "input exceeds maximum context length"
}
```

**With `truncate=false` on nomic-embed-text:**
```
Input: 2,600 chars (exceeds 510 token limit)

Response: HTTP 500
{
  "error": "do embedding request: EOF"
}
```

**Conclusion:**
- `truncate=false` works correctly for all-minilm (gives proper error)
- `truncate=false` doesn't help nomic-embed-text (still gets EOF)
- Different models have **different failure behaviors**

---

## Root Cause

The limits are **NOT in the model files or Ollama configuration** - they appear to be imposed by the **llama.cpp embedding implementation**:

**Evidence:**
- nomic-embed-text GGUF metadata: `context_length: 2048`
- nomic-embed-text Ollama Modelfile: `num_ctx: 8192`
- **Actual embedding limit: 510 tokens** (1/4 of what model file claims!)

**Conclusion:**
1. The limit is **not** in the GGUF model file (model says 2048, actual is 510)
2. The limit is **not** controlled by Ollama's `num_ctx` parameter (Modelfile says 8192)
3. The limit appears to be in the **llama.cpp embedding backend** itself
4. Each model hits different limits (510 for nomic, 254 for all-minilm)
5. The embedding backend closes the connection when limit exceeded (EOF error)

---

## Recommended Solution

### 1. Switch to `/api/embed` Endpoint

Update `embeddings.py` to use the new endpoint and extract `prompt_eval_count`.

### 2. Implement Adaptive Limit Discovery

```python
class OllamaEmbedding:
    def __init__(self, host, model):
        self.host = host
        self.model = model
        self.model_limit = None  # Discovered on first call

    def _discover_limit(self) -> int:
        """Binary search to find model's token limit."""
        low, high = 100, 1000
        last_success = low

        while low <= high:
            mid = (low + high) // 2
            test_text = "word " * mid

            success, result = self._try_embed(test_text)
            if success:
                last_success = result['prompt_eval_count']
                low = mid + 1
            else:
                high = mid - 1

        return last_success

    def embed(self, text: str) -> List[float]:
        # Discover limit once
        if self.model_limit is None:
            self.model_limit = self._discover_limit()

        # Pre-truncate with safety margin
        estimated_tokens = len(text) / 4
        if estimated_tokens > self.model_limit * 0.95:
            safe_chars = int(self.model_limit * 4 * 0.90)
            text = text[:safe_chars]

        # Call API
        result = self._embed_request(text)

        # Verify no unexpected truncation
        actual_tokens = result.get('prompt_eval_count', 0)
        expected_tokens = len(text) / 4

        if actual_tokens < expected_tokens * 0.80:
            # Silent truncation detected, update limit
            self.model_limit = actual_tokens

        return result['embeddings'][0]
```

### 3. Cache Discovered Limits

Store discovered limits in a JSON file:

```json
{
  "nomic-embed-text": 510,
  "all-minilm": 254,
  "mxbai-embed-large": 512
}
```

Load on startup, persist when new models discovered.

### 4. Remove Hardcoded MAX_CHARS

Remove all `MAX_CHARS = 900` constants from:
- `src/cicada_vector/rag.py`
- `src/cicada_vector/git_indexer.py`

Replace with dynamic truncation based on discovered limits.

---

## Implementation Checklist

- [ ] Update `embeddings.py` to use `/api/embed` endpoint
- [ ] Add `_discover_limit()` method with binary search
- [ ] Add `prompt_eval_count` validation
- [ ] Implement limit caching to `~/.cicada/ollama_limits.json`
- [ ] Remove `MAX_CHARS` constants from rag.py and git_indexer.py
- [ ] Update CLAUDE.md with new architecture
- [ ] Add tests for limit discovery and truncation detection
- [ ] Add warning logs when text is truncated

---

## Testing Results

### nomic-embed-text
```
Exact limit: 510 tokens
Character limit: ~2,550 chars (at 5 chars/token)
Failure mode: HTTP 500 EOF error
Detectable: Yes (error thrown)
```

### all-minilm
```
Exact limit: 254 tokens
Character limit: ~3,000 chars (before silent truncation)
Failure mode: Silent truncation (no error)
Detectable: Yes (via prompt_eval_count mismatch)
```

---

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama Embeddings Guide](https://docs.ollama.com/capabilities/embeddings)
- [Embedding Models Blog](https://ollama.com/blog/embedding-models)

---

## Appendix: Why 900 Characters Was Chosen

The original `MAX_CHARS = 900` was likely chosen as:
- Conservative estimate for ~225 tokens (at 4 chars/token)
- Well below nomic-embed-text's 510 token limit
- Safe margin to avoid 500 errors

However, this wastes 56% of available context (510 - 225 = 285 unused tokens).
