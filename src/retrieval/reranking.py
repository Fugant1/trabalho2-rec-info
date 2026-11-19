"""Neural reranking module using a Sentence Transformers Cross-Encoder.

Implements Requirement 9 of Trabalho Pratico 2:
- Takes the hybrid (RRF) ranking as the initial retriever.
- Reorders the Top-50 documents jointly scoring ``(query, document)`` pairs
  with a pre-trained cross-encoder (``cross-encoder/ms-marco-MiniLM-L6-v2``
  by default).
- Keeps documents originally below position 50 in their exact relative order
  from the hybrid ranking (with monotonically compatible tail scores so
  score-based evaluators like ``pytrec_eval`` agree with list order).
- Records latency, model version, max input sequence length (in tokens), and
  whether document/pair truncation occurred.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def _format_corpus_text(raw_doc: Any) -> str:
    if isinstance(raw_doc, Mapping):
        title = str(raw_doc.get("title") or "").strip()
        text = str(raw_doc.get("text") or "").strip()
        return f"{title} {text}".strip() if (title and text) else (title or text)
    return str(raw_doc).strip()


def _normalize_ranking_pairs(raw_ranking: Any) -> list[tuple[str, float]]:
    if raw_ranking is None:
        return []
    if isinstance(raw_ranking, Mapping):
        ordered = sorted(
            ((str(d), float(s)) for d, s in raw_ranking.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ordered

    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for entry in raw_ranking:
        if isinstance(entry, Mapping):
            doc_id = entry.get("doc_id") or entry.get("_id") or entry.get("id")
            if doc_id is None:
                continue
            doc_str = str(doc_id)
            score = float(entry.get("score", 0.0))
        elif isinstance(entry, tuple) and len(entry) >= 2:
            doc_str = str(entry[0])
            score = float(entry[1])
        else:
            raise TypeError(f"Unsupported ranking entry: {type(entry).__name__}")
        if doc_str not in seen:
            seen.add(doc_str)
            out.append((doc_str, score))
    return out


class CrossEncoderReranker:
    """Reorders the top-N hybrid candidates with a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        rerank_top_k: int = 50,
        batch_size: int = 32,
        max_length: int | None = None,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if rerank_top_k <= 0:
            raise ValueError("rerank_top_k must be a positive integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.model_name = model_name
        self.rerank_top_k = int(rerank_top_k)
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.device = device
        self._model = model

        self.last_timing_stats: dict[str, Any] = {}
        self.last_truncation_stats: dict[str, Any] = {}

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model

    def _resolve_max_length(self, model: Any) -> int | None:
        if self.max_length is not None:
            return int(self.max_length)
        for attr in ("max_length",):
            val = getattr(model, attr, None)
            if isinstance(val, int) and 0 < val < 100_000:
                return val
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            val = getattr(tokenizer, "model_max_length", None)
            if isinstance(val, int) and 0 < val < 100_000:
                return val
        inner_model = getattr(model, "model", None)
        config = getattr(inner_model, "config", None)
        val = getattr(config, "max_position_embeddings", None)
        if isinstance(val, int) and 0 < val < 100_000:
            return val
        return None

    @staticmethod
    def _count_truncated_pairs(
        model: Any,
        pairs: Sequence[tuple[str, str]],
        max_length: int | None,
    ) -> int | None:
        """Count ``(query, doc)`` pairs whose tokenized length exceeds ``max_length``."""
        if not pairs or not max_length or max_length <= 0:
            return 0 if (max_length and max_length > 0) else None
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            queries = [q for q, _ in pairs]
            docs = [d for _, d in pairs]
            encoded = tokenizer(
                queries,
                docs,
                truncation=False,
                add_special_tokens=True,
                verbose=False,
            )
            input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
            if input_ids is None:
                return None
            return sum(1 for ids in input_ids if len(ids) > max_length)
        except Exception:
            return None

    @staticmethod
    def _adjust_tail_scores(
        reranked_head: list[tuple[str, float]],
        tail: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Preserve exact relative order of tail docs while keeping scores < head scores."""
        if not tail or not reranked_head:
            return list(tail)

        min_head_score = min(score for _, score in reranked_head)
        max_tail_score = max(score for _, score in tail)
        if max_tail_score < min_head_score:
            return list(tail)

        offset = (min_head_score - max_tail_score) - 1e-4
        return [(doc_id, float(score) + offset) for doc_id, score in tail]

    def rerank_query(
        self,
        query: str,
        initial_ranking: Any,
        corpus_texts: Mapping[str, str],
        *,
        count_truncation: bool = False,
    ) -> tuple[list[tuple[str, float]], dict[str, int | None]]:
        """Rerank top-``rerank_top_k`` documents for one query, preserving tail order."""
        ranked = _normalize_ranking_pairs(initial_ranking)
        if not ranked:
            return [], {"pairs_evaluated": 0, "truncated_pairs": 0}

        head = ranked[: self.rerank_top_k]
        tail = ranked[self.rerank_top_k :]

        pairs = [(query, corpus_texts.get(doc_id, "")) for doc_id, _ in head]
        model = self._get_model()
        max_len = self._resolve_max_length(model)

        truncated: int | None = None
        if count_truncation:
            truncated = self._count_truncated_pairs(model, pairs, max_len)

        raw_scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scored_head = [
            (doc_id, float(score), orig_idx)
            for orig_idx, ((doc_id, _), score) in enumerate(zip(head, raw_scores))
        ]
        # Sort descending by cross-encoder score; break ties by original hybrid rank
        scored_head.sort(key=lambda item: (-item[1], item[2]))
        reranked_head = [(doc_id, score) for doc_id, score, _ in scored_head]

        adjusted_tail = self._adjust_tail_scores(reranked_head, tail)
        return reranked_head + adjusted_tail, {
            "pairs_evaluated": len(pairs),
            "truncated_pairs": truncated,
        }

    def rerank(
        self,
        queries: Mapping[str, str],
        initial_rankings: Mapping[str, Any],
        corpus: Mapping[str, Any],
        *,
        track_truncation: bool = True,
    ) -> dict[str, list[tuple[str, float]]]:
        """Rerank all queries, recording timing and cross-encoder truncation metrics."""
        corpus_texts: dict[str, str] = {
            str(doc_id): _format_corpus_text(raw_doc)
            for doc_id, raw_doc in corpus.items()
        }

        # Ensure model is loaded before timing query reranking
        model = self._get_model()
        max_len = self._resolve_max_length(model)

        qids = [str(qid) for qid in queries.keys()]
        reranked_results: dict[str, list[tuple[str, float]]] = {}

        total_pairs = 0
        total_truncated = 0
        truncation_available = track_truncation

        # Measure reranking inference time cleanly (excluding optional truncation audit)
        pairs_by_qid: dict[str, list[tuple[str, str]]] = {}
        start = time.perf_counter()
        for qid in qids:
            qtext = str(queries[qid])
            initial = initial_rankings.get(qid, [])
            reranked_list, stats = self.rerank_query(
                qtext,
                initial,
                corpus_texts,
                count_truncation=False,
            )
            reranked_results[qid] = reranked_list
            total_pairs += int(stats["pairs_evaluated"] or 0)
            if track_truncation:
                head_docs = _normalize_ranking_pairs(initial)[: self.rerank_top_k]
                pairs_by_qid[qid] = [
                    (qtext, corpus_texts.get(doc_id, "")) for doc_id, _ in head_docs
                ]
        elapsed = time.perf_counter() - start

        if track_truncation:
            for pairs in pairs_by_qid.values():
                trunc = self._count_truncated_pairs(model, pairs, max_len)
                if trunc is None:
                    truncation_available = False
                    break
                total_truncated += trunc

        num_queries = len(qids)
        mean_sec = (elapsed / num_queries) if num_queries > 0 else 0.0
        self.last_timing_stats = {
            "num_queries": num_queries,
            "rerank_top_k": self.rerank_top_k,
            "batch_size": self.batch_size,
            "total_query_time_seconds": elapsed,
            "mean_query_time_seconds": mean_sec,
            "mean_query_time_ms": mean_sec * 1000.0,
        }
        self.last_truncation_stats = {
            "max_length": max_len,
            "total_pairs_evaluated": total_pairs,
            "truncated_pairs_count": total_truncated if truncation_available else None,
            "truncated_pairs_ratio": (
                (total_truncated / total_pairs)
                if (truncation_available and total_pairs > 0)
                else None
            ),
            "had_truncation": (
                total_truncated > 0 if truncation_available else None
            ),
        }
        return reranked_results

    def get_metadata(self) -> dict[str, Any]:
        try:
            import sentence_transformers

            library_version = f"sentence-transformers=={sentence_transformers.__version__}"
        except Exception:
            library_version = "sentence-transformers"

        return {
            "model_name": self.model_name,
            "library": library_version,
            "rerank_top_k": self.rerank_top_k,
            "batch_size": self.batch_size,
            "truncation": self.last_truncation_stats,
            "timing": self.last_timing_stats,
        }
