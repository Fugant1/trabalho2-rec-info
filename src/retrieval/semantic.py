"""Semantic dense retrieval using precomputed bi-encoder document embeddings.

Implements Requirement 5 of Trabalho Pratico 2:
- Reuses precomputed document embeddings cached by :class:`EmbeddingIndexer`.
- Encodes queries with the same Sentence Transformers bi-encoder model.
- Performs exhaustive cosine similarity search over the 3,633 NFCorpus docs.
- Records embedding metadata, truncation info, and query processing times.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.pre_processing.semantic_index import DEFAULT_EMBEDDING_MODEL


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)


class SemanticRetriever:
    """Dense bi-encoder retriever computing cosine similarity over stored embeddings."""

    def __init__(
        self,
        embeddings: np.ndarray,
        doc_ids: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
        model_name: str | None = None,
        normalize_embeddings: bool | None = None,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"embeddings must be a 2D array, got shape {arr.shape}.")
        if arr.shape[0] != len(doc_ids):
            raise ValueError(
                f"embeddings rows ({arr.shape[0]}) != len(doc_ids) ({len(doc_ids)})."
            )

        self.metadata: dict[str, Any] = dict(metadata or {})
        self.doc_ids: list[str] = [str(d) for d in doc_ids]
        self._doc_id_to_idx: dict[str, int] = {
            doc_id: idx for idx, doc_id in enumerate(self.doc_ids)
        }

        self.model_name = (
            model_name
            or self.metadata.get("model_name")
            or DEFAULT_EMBEDDING_MODEL
        )
        self.normalize_embeddings = (
            bool(normalize_embeddings)
            if normalize_embeddings is not None
            else bool(self.metadata.get("normalized", True))
        )
        self.device = device
        self._model = model

        # Always store L2-normalized document embeddings so dot product == cosine similarity
        self.doc_embeddings: np.ndarray = (
            arr if self.normalize_embeddings else _l2_normalize(arr)
        )
        self.last_timing_stats: dict[str, Any] = {}

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def encode_queries(
        self,
        queries: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode one or more query strings into L2-normalized float32 embeddings."""
        model = self._get_model()
        vecs = model.encode(
            list(queries),
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]
        if not self.normalize_embeddings:
            vecs = _l2_normalize(vecs)
        return vecs

    def _rank_scores(self, scores: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        num_docs = len(self.doc_ids)
        k = min(top_k, num_docs)
        if k <= 0:
            return []

        if k < num_docs:
            # Fast partial selection followed by exact deterministic sort
            top_indices = np.argpartition(-scores, kth=k - 1)[:k]
        else:
            top_indices = np.arange(num_docs)

        pairs = [(self.doc_ids[int(idx)], float(scores[int(idx)])) for idx in top_indices]
        pairs.sort(key=lambda item: (-item[1], item[0]))
        return pairs[:k]

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Retrieve top-k documents for a single query by cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not self.doc_ids:
            return []

        q_vec = self.encode_queries([query], batch_size=1)[0]
        scores = self.doc_embeddings @ q_vec
        return self._rank_scores(scores, top_k=top_k)

    def retrieve(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
        batch_size: int | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Retrieve top-k documents for all queries and record processing times.

        When ``batch_size`` is ``None``, queries are encoded and scored
        individually so ``mean_query_time_seconds`` reflects true per-query
        latency. Pass an integer ``batch_size`` to encode queries in batches.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        qids = [str(qid) for qid in queries.keys()]
        qtexts = [str(queries[qid]) for qid in queries.keys()]

        # Ensure model is loaded before timing query execution (load cost != query latency)
        _ = self._get_model()

        rankings: dict[str, list[tuple[str, float]]] = {}
        start = time.perf_counter()

        if not qids:
            elapsed = 0.0
        elif batch_size is None:
            for qid, qtext in zip(qids, qtexts):
                q_vec = self.encode_queries([qtext], batch_size=1)[0]
                scores = self.doc_embeddings @ q_vec
                rankings[qid] = self._rank_scores(scores, top_k=top_k)
            elapsed = time.perf_counter() - start
        else:
            q_matrix = self.encode_queries(qtexts, batch_size=batch_size)
            sim_matrix = q_matrix @ self.doc_embeddings.T
            for row_idx, qid in enumerate(qids):
                rankings[qid] = self._rank_scores(sim_matrix[row_idx], top_k=top_k)
            elapsed = time.perf_counter() - start

        num_queries = len(qids)
        mean_sec = (elapsed / num_queries) if num_queries > 0 else 0.0
        self.last_timing_stats = {
            "num_queries": num_queries,
            "top_k": top_k,
            "batch_size": batch_size,
            "total_query_time_seconds": elapsed,
            "mean_query_time_seconds": mean_sec,
            "mean_query_time_ms": mean_sec * 1000.0,
        }
        return rankings

    def score_pair(self, query: str, doc_id: str) -> float | None:
        """Compute exact cosine similarity between ``query`` and a specific ``doc_id``."""
        idx = self._doc_id_to_idx.get(str(doc_id))
        if idx is None:
            return None
        q_vec = self.encode_queries([query], batch_size=1)[0]
        return float(np.dot(self.doc_embeddings[idx], q_vec))

    def get_metadata(self) -> dict[str, Any]:
        """Return all required semantic retrieval metadata (Requirement 5)."""
        return {
            "model_name": self.model_name,
            "library": self.metadata.get("library", "sentence-transformers"),
            "embedding_dim": int(self.doc_embeddings.shape[1]),
            "num_documents": len(self.doc_ids),
            "similarity_function": "cosine",
            "normalized": self.normalize_embeddings,
            "max_seq_length": self.metadata.get("max_seq_length"),
            "truncated_docs_count": self.metadata.get("truncated_docs_count"),
            "had_truncation": self.metadata.get("had_truncation"),
            "doc_embedding_time_seconds": self.metadata.get("build_time_seconds"),
            "timing": self.last_timing_stats,
        }
