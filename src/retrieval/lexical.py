"""Lexical retrieval baseline using Okapi BM25 over the in-house inverted index.

Implements Requirements 2 and 3 of Trabalho Pratico 2:
- Uses ONLY the inverted index posting lists of query terms (never iterates
  over the entire corpus per query).
- Applies the exact same tokenization / stopword / normalization parameters
  used when building the inverted index.
- Tracks mean query processing latency and exposes term-level match evidence
  to support Requirement 7 (lexical vs. semantic error/advantage analysis).
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class BM25Retriever:
    """Okapi BM25 retriever operating strictly on inverted index posting lists."""

    def __init__(
        self,
        lexical_index: Mapping[str, Any],
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Iterable[str] | None = None,
    ) -> None:
        if k1 < 0.0:
            raise ValueError("k1 must be non-negative.")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0.0 and 1.0.")

        self.k1 = float(k1)
        self.b = float(b)

        self.N = int(lexical_index["N"])
        self.avg_doc_length = float(lexical_index["avg_doc_length"])
        self.doc_lengths: dict[str, int] = {
            str(doc_id): int(length)
            for doc_id, length in lexical_index["doc_lengths"].items()
        }
        self.index: dict[str, dict[str, int]] = lexical_index["index"]
        self.df: dict[str, int] = (
            {str(t): int(c) for t, c in lexical_index["df"].items()}
            if "df" in lexical_index
            else {term: len(postings) for term, postings in self.index.items()}
        )
        self.vocab_size = int(lexical_index.get("vocab_size", len(self.index)))
        self.total_postings = int(
            lexical_index.get("total_postings", sum(self.df.values()))
        )
        self.build_time_seconds = lexical_index.get("build_time_seconds")

        params = lexical_index.get("params") or {}
        self.lowercase = bool(params.get("lowercase", True))
        self.remove_stopwords = bool(params.get("remove_stopwords", True))
        self.min_token_length = int(params.get("min_token_length", 1))
        self.token_pattern = str(params.get("token_pattern", r"[a-z0-9]+"))
        self._regex = re.compile(self.token_pattern)
        self._stopwords = (
            frozenset(w.lower() for w in stopwords)
            if stopwords is not None
            else ENGLISH_STOP_WORDS
        )

        # Precompute Robertson-Sparck Jones BM25 IDF (non-negative Lucene variant)
        self.idf: dict[str, float] = {
            term: math.log(((self.N - df_val + 0.5) / (df_val + 0.5)) + 1.0)
            for term, df_val in self.df.items()
        }

        self.last_timing_stats: dict[str, Any] = {}

    def tokenize_query(self, query: str) -> list[str]:
        """Tokenize a query with the exact parameters stored in the inverted index."""
        normalized = query.lower() if self.lowercase else query
        tokens = self._regex.findall(normalized)
        if self.min_token_length > 1:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]
        if self.remove_stopwords:
            stop = self._stopwords
            tokens = [t for t in tokens if t.lower() not in stop]
        return tokens

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Rank matching documents for a single query using posting lists only."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        query_tokens = self.tokenize_query(query)
        if not query_tokens or self.N == 0 or self.avg_doc_length <= 0.0:
            return []

        query_term_counts = Counter(query_tokens)
        scores: dict[str, float] = {}

        k1 = self.k1
        b = self.b
        avgdl = self.avg_doc_length
        doc_lengths = self.doc_lengths
        index = self.index
        idf_map = self.idf

        # Term-At-A-Time accumulation strictly over matching posting lists
        for term, qtf in query_term_counts.items():
            postings = index.get(term)
            if not postings:
                continue
            term_idf = idf_map[term]
            weight = qtf * term_idf
            for doc_id, tf in postings.items():
                dl = doc_lengths[doc_id]
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                scores[doc_id] = scores.get(doc_id, 0.0) + weight * (
                    (tf * (k1 + 1.0)) / denom
                )

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def retrieve(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Retrieve top-k rankings for a batch of queries and record latency."""
        rankings: dict[str, list[tuple[str, float]]] = {}
        start = time.perf_counter()
        for qid, qtext in queries.items():
            rankings[str(qid)] = self.search(str(qtext), top_k=top_k)
        elapsed = time.perf_counter() - start

        num_queries = len(queries)
        mean_sec = (elapsed / num_queries) if num_queries > 0 else 0.0
        self.last_timing_stats = {
            "num_queries": num_queries,
            "top_k": top_k,
            "total_query_time_seconds": elapsed,
            "mean_query_time_seconds": mean_sec,
            "mean_query_time_ms": mean_sec * 1000.0,
        }
        return rankings

    def explain_doc_match(self, query: str, doc_id: str) -> dict[str, Any]:
        """Return concrete term-level BM25 evidence for a ``(query, doc_id)`` pair."""
        doc_key = str(doc_id)
        query_tokens = self.tokenize_query(query)
        unique_terms = list(dict.fromkeys(query_tokens))
        qtf_map = Counter(query_tokens)
        dl = self.doc_lengths.get(doc_key, 0)

        matched_terms: list[dict[str, Any]] = []
        missing_terms: list[str] = []
        total_score = 0.0

        for term in unique_terms:
            postings = self.index.get(term)
            tf = postings.get(doc_key, 0) if postings else 0
            df_val = self.df.get(term, 0)
            idf_val = self.idf.get(term, 0.0)
            if tf > 0 and self.avg_doc_length > 0:
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * (dl / self.avg_doc_length)
                )
                contrib = (
                    qtf_map[term]
                    * idf_val
                    * ((tf * (self.k1 + 1.0)) / denom)
                )
                total_score += contrib
                matched_terms.append(
                    {
                        "term": term,
                        "qtf": qtf_map[term],
                        "tf": tf,
                        "df": df_val,
                        "idf": idf_val,
                        "bm25_contribution": contrib,
                    }
                )
            else:
                missing_terms.append(term)

        overlap_ratio = (
            len(matched_terms) / len(unique_terms) if unique_terms else 0.0
        )
        return {
            "doc_id": doc_key,
            "doc_length_tokens": dl,
            "avg_doc_length_tokens": self.avg_doc_length,
            "query_tokens": query_tokens,
            "matched_terms": matched_terms,
            "missing_terms": missing_terms,
            "lexical_overlap_ratio": overlap_ratio,
            "bm25_score": total_score,
        }

    def get_metadata(self) -> dict[str, Any]:
        """Return lexical index & BM25 configuration metadata for reporting."""
        return {
            "model": "BM25",
            "k1": self.k1,
            "b": self.b,
            "N": self.N,
            "vocab_size": self.vocab_size,
            "total_postings": self.total_postings,
            "avg_doc_length": self.avg_doc_length,
            "build_time_seconds": self.build_time_seconds,
            "preprocessing": {
                "lowercase": self.lowercase,
                "remove_stopwords": self.remove_stopwords,
                "min_token_length": self.min_token_length,
                "token_pattern": self.token_pattern,
            },
            "timing": self.last_timing_stats,
        }

    @classmethod
    def tune_on_dev(
        cls,
        lexical_index: Mapping[str, Any],
        dev_queries: Mapping[str, str],
        dev_qrels: Mapping[str, Mapping[str, int | float]],
        evaluator: Any,
        k1_values: Sequence[float] = (1.2, 1.5, 1.8),
        b_values: Sequence[float] = (0.5, 0.75, 0.9),
        top_k: int = 100,
    ) -> dict[str, Any]:
        """Tune BM25 ``(k1, b)`` strictly on the ``dev`` split by maximizing NDCG@10."""
        subset_queries = {
            str(qid): str(dev_queries[qid])
            for qid in sorted(dev_qrels.keys())
            if qid in dev_queries
        }
        grid_results: list[dict[str, Any]] = []
        best_entry: dict[str, Any] | None = None

        for k1 in k1_values:
            for b in b_values:
                retriever = cls(lexical_index, k1=k1, b=b)
                rankings = retriever.retrieve(subset_queries, top_k=top_k)
                eval_res = evaluator.evaluate(dev_qrels, rankings, k=10, model_name="bm25_dev")
                entry = {
                    "k1": float(k1),
                    "b": float(b),
                    "global_metrics": eval_res["global"],
                    "ndcg10": eval_res["global"]["NDCG@10"],
                }
                grid_results.append(entry)
                if best_entry is None or entry["ndcg10"] > best_entry["ndcg10"]:
                    best_entry = entry

        assert best_entry is not None
        return {
            "best_k1": best_entry["k1"],
            "best_b": best_entry["b"],
            "best_ndcg10": best_entry["ndcg10"],
            "grid_results": grid_results,
        }


# Convenient alias matching module naming
LexicalRetriever = BM25Retriever
