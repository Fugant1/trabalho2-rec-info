"""Hybrid retrieval using Reciprocal Rank Fusion (RRF).

Implements Requirement 8 of Trabalho Pratico 2:
- Fuses the top 100 documents from the lexical and semantic rankings using
  ``RRF(d) = sum_s 1 / (k + rank_s(d))``, where ``rank_s(d)`` starts at 1
  and absent documents contribute 0.
- Tunes ``k`` strictly on the ``dev`` split across at least ``{10, 60, 100}``,
  selecting the value with the highest mean ``NDCG@10`` on ``dev``.
- Preserves deterministic ordering and records fusion timing statistics.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any


def _extract_ordered_doc_ids(raw_ranking: Any, limit: int) -> list[str]:
    """Extract up to ``limit`` deduplicated doc_ids preserving ranking order."""
    if raw_ranking is None:
        return []

    items: list[str] = []
    if isinstance(raw_ranking, Mapping):
        ordered = sorted(
            ((str(d), float(s)) for d, s in raw_ranking.items()),
            key=lambda pair: (-pair[1], pair[0]),
        )
        items = [doc_id for doc_id, _ in ordered]
    elif isinstance(raw_ranking, Sequence) and not isinstance(raw_ranking, (str, bytes)):
        for entry in raw_ranking:
            if isinstance(entry, Mapping):
                doc_id = entry.get("doc_id") or entry.get("_id") or entry.get("id")
                if doc_id is not None:
                    items.append(str(doc_id))
            elif isinstance(entry, tuple) and len(entry) >= 1:
                items.append(str(entry[0]))
            elif isinstance(entry, str):
                items.append(entry)
    else:
        raise TypeError(f"Unsupported ranking container: {type(raw_ranking).__name__}")

    seen: set[str] = set()
    deduped: list[str] = []
    for doc_id in items:
        if doc_id not in seen:
            seen.add(doc_id)
            deduped.append(doc_id)
            if len(deduped) >= limit:
                break
    return deduped


def reciprocal_rank_fusion(
    rankings_list: Sequence[Any],
    k: int = 60,
    fusion_depth: int = 100,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple rankings for a single query with Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings_list:
        Sequence of rankings (one per retrieval system) for the same query.
    k:
        RRF smoothing constant ``k > 0``.
    fusion_depth:
        Maximum number of top documents taken from each system (default ``100``).
    top_k:
        Optional cutoff for the returned fused ranking (``None`` returns all
        unique fused documents).
    """
    if k <= 0:
        raise ValueError("RRF parameter k must be a positive integer.")
    if fusion_depth <= 0:
        raise ValueError("fusion_depth must be a positive integer.")

    rrf_scores: dict[str, float] = {}
    for ranking in rankings_list:
        doc_ids = _extract_ordered_doc_ids(ranking, limit=fusion_depth)
        for rank, doc_id in enumerate(doc_ids, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    fused = sorted(rrf_scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        return fused[:top_k]
    return fused


class HybridRetriever:
    """Combines lexical and semantic rankings via Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        k: int = 60,
        fusion_depth: int = 100,
    ) -> None:
        if k <= 0:
            raise ValueError("RRF parameter k must be a positive integer.")
        if fusion_depth <= 0:
            raise ValueError("fusion_depth must be a positive integer.")

        self.k = int(k)
        self.fusion_depth = int(fusion_depth)
        self.last_timing_stats: dict[str, Any] = {}
        self.dev_tuning_report: dict[str, Any] | None = None

    def fuse_query(
        self,
        lexical_ranking: Any,
        semantic_ranking: Any,
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Fuse lexical and semantic rankings for a single query."""
        return reciprocal_rank_fusion(
            [lexical_ranking, semantic_ranking],
            k=self.k,
            fusion_depth=self.fusion_depth,
            top_k=top_k,
        )

    def fuse(
        self,
        lexical_rankings: Mapping[str, Any],
        semantic_rankings: Mapping[str, Any],
        query_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """Fuse lexical and semantic rankings across all queries and record timing."""
        if query_ids is not None:
            qids = [str(q) for q in query_ids]
        else:
            qids = sorted(
                set(str(q) for q in lexical_rankings.keys())
                | set(str(q) for q in semantic_rankings.keys())
            )

        fused_rankings: dict[str, list[tuple[str, float]]] = {}
        start = time.perf_counter()
        for qid in qids:
            fused_rankings[qid] = self.fuse_query(
                lexical_rankings.get(qid, []),
                semantic_rankings.get(qid, []),
                top_k=top_k,
            )
        elapsed = time.perf_counter() - start

        num_queries = len(qids)
        mean_sec = (elapsed / num_queries) if num_queries > 0 else 0.0
        self.last_timing_stats = {
            "num_queries": num_queries,
            "k": self.k,
            "fusion_depth": self.fusion_depth,
            "total_query_time_seconds": elapsed,
            "mean_query_time_seconds": mean_sec,
            "mean_query_time_ms": mean_sec * 1000.0,
        }
        return fused_rankings

    def tune_k(
        self,
        lexical_dev_rankings: Mapping[str, Any],
        semantic_dev_rankings: Mapping[str, Any],
        dev_qrels: Mapping[str, Mapping[str, int | float]],
        evaluator: Any,
        k_candidates: Sequence[int] = (10, 60, 100),
        fusion_depth: int | None = None,
    ) -> dict[str, Any]:
        """Tune RRF ``k`` on the ``dev`` split by maximizing mean ``NDCG@10``.

        Updates ``self.k`` to the winning ``k`` and stores the tuning report.
        """
        if not k_candidates:
            raise ValueError("k_candidates cannot be empty.")

        depth = self.fusion_depth if fusion_depth is None else int(fusion_depth)
        dev_qids = sorted(str(q) for q in dev_qrels.keys())

        results_by_k: dict[int, dict[str, Any]] = {}
        best_k: int | None = None
        best_ndcg: float = -1.0

        for candidate_k in k_candidates:
            k_val = int(candidate_k)
            if k_val <= 0:
                raise ValueError(f"Invalid RRF k candidate: {k_val}")
            fused_dev = {
                qid: reciprocal_rank_fusion(
                    [
                        lexical_dev_rankings.get(qid, []),
                        semantic_dev_rankings.get(qid, []),
                    ],
                    k=k_val,
                    fusion_depth=depth,
                )
                for qid in dev_qids
            }
            eval_res = evaluator.evaluate(
                dev_qrels,
                fused_dev,
                k=10,
                model_name=f"hybrid_rrf_k{k_val}_dev",
            )
            ndcg10 = float(eval_res["global"]["NDCG@10"])
            results_by_k[k_val] = {
                "k": k_val,
                "fusion_depth": depth,
                "global_metrics": eval_res["global"],
                "NDCG@10": ndcg10,
            }
            if best_k is None or ndcg10 > best_ndcg:
                best_k = k_val
                best_ndcg = ndcg10

        assert best_k is not None
        self.k = best_k
        self.fusion_depth = depth
        self.dev_tuning_report = {
            "best_k": best_k,
            "best_dev_ndcg10": best_ndcg,
            "fusion_depth": depth,
            "k_candidates": [int(c) for c in k_candidates],
            "results_by_k": {str(k): v for k, v in results_by_k.items()},
        }
        return self.dev_tuning_report

    def get_metadata(self) -> dict[str, Any]:
        return {
            "method": "Reciprocal Rank Fusion (RRF)",
            "k": self.k,
            "fusion_depth": self.fusion_depth,
            "dev_tuning": self.dev_tuning_report,
            "timing": self.last_timing_stats,
        }
