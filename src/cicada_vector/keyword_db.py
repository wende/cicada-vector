"""
Simple Inverted Index for exact keyword matching.
"""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Set

class KeywordDB:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        # Map: keyword -> Set[doc_id]
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self._dirty = False
        
        if os.path.exists(storage_path):
            self._load()

    def _tokenize(self, text: str) -> Set[str]:
        """Simple tokenization: lowercase, split by non-alphanumeric."""
        # This keeps 'snake_case', 'camelCase', 'numbers123' as distinct tokens
        # but strips punctuation like .,;()
        return set(word.lower() for word in re.findall(r"\b\w+\b", text))

    def _load(self):
        """Load index from JSON file."""
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert lists back to sets and restore defaultdict
            self.index = defaultdict(set, {k: set(v) for k, v in data.items()})

    def add(self, id: str, text: str):
        """Add document text to index."""
        tokens = self._tokenize(text)
        for token in tokens:
            self.index[token].add(id)
        self._dirty = True

    def persist(self):
        """Save index to disk."""
        if not self._dirty:
            return
        
        # Convert sets to lists for JSON serialization
        serializable = {k: list(v) for k, v in self.index.items()}
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f)
        self._dirty = False

    def search(self, query: str) -> List[str]:
        """
        Return list of doc_ids that contain the query terms.
        For multiple words, this implements an OR search (matches any word).
        """
        tokens = self._tokenize(query)
        results = set()
        for token in tokens:
            if token in self.index:
                results.update(self.index[token])
        return list(results)
