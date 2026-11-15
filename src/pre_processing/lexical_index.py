"""Lexical inverted index builder for BM25 / TF-IDF retrieval.

Implements an in-house inverted index from scratch (Requirement 2 of
Trabalho Pratico 2) storing term frequencies (TF), document frequencies (DF),
document lengths, global corpus statistics (N, avg_doc_length), vocabulary size,
total postings, and build time.

Intended to be used solely by :class:`src.pre_processing.pre_processor.PreProcessor`.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class InvertedIndexer:
    """Builds a clean, BM25-ready inverted index from preprocessed tokens."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = True,
        min_token_length: int = 1,
        token_pattern: str = r"[a-z0-9]+",
        stopwords: Iterable[str] | None = None,
    ) -> None:
        if min_token_length < 1:
            raise ValueError("min_token_length must be at least 1.")

        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.token_pattern = token_pattern
        self._regex = re.compile(token_pattern)
        self._stopwords = (
            frozenset(w.lower() for w in stopwords)
            if stopwords is not None
            else ENGLISH_STOP_WORDS
        )

        # term -> {doc_id: tf}
        self.index: dict[str, dict[str, int]] = {}
        # doc_id -> token count
        self.doc_lengths: dict[str, int] = {}

    def tokenize(self, text: str) -> list[str]:
        """Tokenize and normalize text according to the configured parameters."""
        normalized = text.lower() if self.lowercase else text
        tokens = self._regex.findall(normalized)
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        if self.remove_stopwords:
            stop = self._stopwords
            tokens = [t for t in tokens if t.lower() not in stop]
        return tokens

    def add_document(self, doc_id: str, terms: list[str]) -> None:
        """Insert a tokenized document into the inverted index."""
        doc_key = str(doc_id)
        self.doc_lengths[doc_key] = len(terms)
        term_counts = Counter(terms)
        for term, tf in term_counts.items():
            postings = self.index.setdefault(term, {})
            postings[doc_key] = tf

    def build(
        self,
        documents: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> dict[str, Any]:
        """Build the inverted index over all documents and return cache payload."""
        self.index.clear()
        self.doc_lengths.clear()

        start = time.perf_counter()
        items = documents.items() if isinstance(documents, Mapping) else documents
        for doc_id, text in items:
            tokens = self.tokenize(text)
            self.add_document(str(doc_id), tokens)
        build_time = time.perf_counter() - start

        num_docs = len(self.doc_lengths)
        avg_doc_length = (
            sum(self.doc_lengths.values()) / num_docs if num_docs > 0 else 0.0
        )
        df = {term: len(postings) for term, postings in self.index.items()}
        vocab_size = len(self.index)
        total_postings = sum(df.values())

        return {
            "N": num_docs,
            "avg_doc_length": avg_doc_length,
            "vocab_size": vocab_size,
            "total_postings": total_postings,
            "build_time_seconds": build_time,
            "params": {
                "lowercase": self.lowercase,
                "remove_stopwords": self.remove_stopwords,
                "min_token_length": self.min_token_length,
                "token_pattern": self.token_pattern,
                "stopwords_source": "custom"
                if self._stopwords is not ENGLISH_STOP_WORDS
                else "sklearn.ENGLISH_STOP_WORDS",
            },
            "doc_lengths": self.doc_lengths,
            "df": df,
            "index": self.index,
        }
