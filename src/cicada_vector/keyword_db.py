"""
Inverted index for keyword matching with IDF-weighted scoring.
"""

import json
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Set


_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further get
got had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into
is isn't it it's its itself let's me more most mustn't my myself no nor not of
off on once only or other ought our ours ourselves out over own same shan't she
she'd she'll she's should shouldn't so some such than that that's the their
theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up upon very was wasn't we we'd
we'll we're we've were weren't what what's when when's where where's which
while who who's whom why why's will with won't would wouldn't you you'd you'll
you're you've your yours yourself yourselves also just like one two three four
five use used using many much new even still already way well make made really
""".split())


class KeywordDB:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        # Map: keyword -> Set[doc_id]
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self._total_docs: int = 0
        self._dirty = False

        if os.path.exists(storage_path) and os.path.getsize(storage_path) > 0:
            self._load()

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize and strip stopwords — keeps proper nouns and domain terms."""
        return set(
            word.lower() for word in re.findall(r"\b\w+\b", text)
        ) - _STOPWORDS

    def _load(self):
        """Load index from JSON file."""
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if '_meta' in data:
            meta = data.pop('_meta')
            self._total_docs = meta.get('total_docs', 0)
        else:
            # Legacy format: compute total_docs from unique doc IDs across all terms
            all_docs: Set[str] = set()
            for docs in data.values():
                all_docs.update(docs)
            self._total_docs = len(all_docs)

        self.index = defaultdict(set, {k: set(v) for k, v in data.items()})

    def add(self, id: str, text: str):
        """Add document text to index."""
        tokens = self._tokenize(text)
        for token in tokens:
            self.index[token].add(id)
        self._total_docs += 1
        self._dirty = True

    def persist(self):
        """Save index to disk."""
        if not self._dirty:
            return

        serializable: dict = {'_meta': {'total_docs': self._total_docs}}
        serializable.update({k: list(v) for k, v in self.index.items()})
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f)
        self._dirty = False

    def search(self, query: str) -> List[str]:
        """
        Return doc_ids ranked by IDF-weighted score.
        Documents matching rarer, more distinctive query terms rank higher.
        """
        tokens = self._tokenize(query)
        if not tokens or self._total_docs == 0:
            return []

        scores: Dict[str, float] = {}
        for token in tokens:
            if token in self.index:
                df = len(self.index[token])
                idf = math.log(1.0 + self._total_docs / df)
                for doc_id in self.index[token]:
                    scores[doc_id] = scores.get(doc_id, 0.0) + idf

        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
