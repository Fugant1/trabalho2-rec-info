"""Unified retrieval orchestrator combining lexical, semantic, hybrid, and reranking."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .hybrid import HybridRetriever
from .lexical import BM25Retriever
from .reranking import CrossEncoderReranker
from .semantic import SemanticRetriever


class Retriever:
    """Facade coordinating the four IR stages: lexical, semantic, hybrid, reranking."""

    def __init__(
        self,
        lexical: BM25Retriever,
        semantic: SemanticRetriever,
        hybrid: HybridRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
        corpus: Mapping[str, Any] | None = None,
    ) -> None:
        self.lexical = lexical
        self.semantic = semantic
        self.hybrid = hybrid or HybridRetriever(k=60, fusion_depth=100)
        self.reranker = reranker or CrossEncoderReranker(rerank_top_k=50)
        self.corpus: dict[str, Any] = dict(corpus) if corpus is not None else {}

    def retrieve_lexical(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        return self.lexical.retrieve(queries, top_k=top_k)

    def retrieve_semantic(
        self,
        queries: Mapping[str, str],
        top_k: int = 100,
        batch_size: int | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        return self.semantic.retrieve(queries, top_k=top_k, batch_size=batch_size)

    def retrieve_hybrid(
        self,
        lexical_rankings: Mapping[str, Any],
        semantic_rankings: Mapping[str, Any],
        top_k: int | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        return self.hybrid.fuse(lexical_rankings, semantic_rankings, top_k=top_k)

    def rerank(
        self,
        queries: Mapping[str, str],
        hybrid_rankings: Mapping[str, Any],
        corpus: Mapping[str, Any] | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        active_corpus = corpus if corpus is not None else self.corpus
        if not active_corpus:
            raise ValueError("A corpus mapping is required for cross-encoder reranking.")
        return self.reranker.rerank(queries, hybrid_rankings, active_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "reranking",
        corpus: Mapping[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Run single-query retrieval under ``lexical``, ``semantic``, ``hybrid``, or ``reranking``."""
        mode_norm = mode.strip().lower()
        if mode_norm == "lexical":
            return self.lexical.search(query, top_k=top_k)
        if mode_norm == "semantic":
            return self.semantic.search(query, top_k=top_k)

        lex_top = self.lexical.search(query, top_k=max(self.hybrid.fusion_depth, top_k))
        sem_top = self.semantic.search(query, top_k=max(self.hybrid.fusion_depth, top_k))
        hyb_top = self.hybrid.fuse_query(lex_top, sem_top)
        if mode_norm == "hybrid":
            return hyb_top[:top_k]

        if mode_norm == "reranking":
            active_corpus = corpus if corpus is not None else self.corpus
            if not active_corpus:
                raise ValueError(
                    "A corpus mapping is required for single-query reranking."
                )
            corpus_texts = {
                str(doc_id): (
                    f"{(doc.get('title') or '').strip()} {(doc.get('text') or '').strip()}".strip()
                    if isinstance(doc, Mapping)
                    else str(doc)
                )
                for doc_id, doc in active_corpus.items()
            }
            reranked, _ = self.reranker.rerank_query(query, hyb_top, corpus_texts)
            return reranked[:top_k]

        raise ValueError(
            f"Unsupported retrieval mode '{mode}'. Expected one of "
            "'lexical', 'semantic', 'hybrid', or 'reranking'."
        )

    def retrieve_all_stages(
        self,
        queries: Mapping[str, str],
        corpus: Mapping[str, Any] | None = None,
        initial_top_k: int = 100,
        semantic_batch_size: int | None = None,
    ) -> dict[str, dict[str, list[tuple[str, float]]]]:
        """Run lexical, semantic, hybrid (RRF), and cross-encoder reranking stages."""
        lexical_rankings = self.retrieve_lexical(queries, top_k=initial_top_k)
        semantic_rankings = self.retrieve_semantic(
            queries,
            top_k=initial_top_k,
            batch_size=semantic_batch_size,
        )
        hybrid_rankings = self.retrieve_hybrid(lexical_rankings, semantic_rankings)
        reranking_rankings = self.rerank(queries, hybrid_rankings, corpus=corpus)
        return {
            "lexical": lexical_rankings,
            "semantic": semantic_rankings,
            "hybrid": hybrid_rankings,
            "reranking": reranking_rankings,
        }

    def get_metadata(self) -> dict[str, Any]:
        return {
            "lexical": self.lexical.get_metadata(),
            "semantic": self.semantic.get_metadata(),
            "hybrid": self.hybrid.get_metadata(),
            "reranking": self.reranker.get_metadata(),
        }
