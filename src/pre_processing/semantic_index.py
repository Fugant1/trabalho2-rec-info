"""Semantic dense embedding indexer using Sentence Transformers.

Implements precomputation of document embeddings (Requirement 5 of
Trabalho Pratico 2) using the bi-encoder model suggested in the specification
(``sentence-transformers/all-MiniLM-L6-v2`` by default).

Intended to be used solely by :class:`src.pre_processing.pre_processor.PreProcessor`.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingIndexer:
    """Computes normalized dense document embeddings and indexing metadata."""

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
        show_progress_bar: bool = False,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.embedding_model_name = embedding_model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        self.show_progress_bar = show_progress_bar
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(self.embedding_model_name, **kwargs)
        return self._model

    @staticmethod
    def _count_truncated_documents(
        model: Any,
        texts: list[str],
        max_seq_length: int | None,
    ) -> int | None:
        """Count documents whose tokenized length exceeds model.max_seq_length."""
        if not max_seq_length or max_seq_length <= 0:
            return None
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return None
        try:
            encoded = tokenizer(
                texts,
                truncation=False,
                add_special_tokens=True,
                verbose=False,
            )
            input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
            if input_ids is None:
                return None
            return sum(1 for ids in input_ids if len(ids) > max_seq_length)
        except Exception:
            return None

    def index(
        self,
        documents: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Encode documents into dense embeddings and return ``(embeddings, metadata)``."""
        items = list(documents.items()) if isinstance(documents, Mapping) else list(documents)
        doc_ids = [str(doc_id) for doc_id, _ in items]
        texts = [str(text) for _, text in items]

        if not doc_ids:
            raise ValueError("Cannot build semantic index from an empty corpus.")

        model = self._get_model()
        max_seq_length = getattr(model, "max_seq_length", None)
        if max_seq_length is not None:
            max_seq_length = int(max_seq_length)

        truncated_docs_count = self._count_truncated_documents(
            model, texts, max_seq_length
        )

        start = time.perf_counter()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )
        build_time = time.perf_counter() - start

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(doc_ids):
            raise ValueError(
                f"Unexpected embeddings shape {embeddings.shape} for "
                f"{len(doc_ids)} documents."
            )

        try:
            import sentence_transformers

            library_version = f"sentence-transformers=={sentence_transformers.__version__}"
        except Exception:
            library_version = "sentence-transformers"

        metadata: dict[str, Any] = {
            "doc_ids": doc_ids,
            "num_documents": len(doc_ids),
            "model_name": self.embedding_model_name,
            "library": library_version,
            "embedding_dim": int(embeddings.shape[1]),
            "similarity_function": "cosine" if self.normalize_embeddings else "dot_or_cosine",
            "normalized": self.normalize_embeddings,
            "batch_size": self.batch_size,
            "max_seq_length": max_seq_length,
            "truncated_docs_count": truncated_docs_count,
            "had_truncation": (
                truncated_docs_count > 0
                if truncated_docs_count is not None
                else None
            ),
            "build_time_seconds": build_time,
        }
        return embeddings, metadata
