"""Information Retrieval and RAG evaluation module.

Implements all quantitative evaluation metrics, per-query comparison helpers,
typo recovery calculations, RAG retrieval/generation evaluation, and
standardized CSV exports required by Trabalho Pratico 2 (Items 4, 6, 7, 10,
12, 13, and 15).
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytrec_eval


def _normalize_qrels(
    qrels: Mapping[str, Mapping[str, int | float]],
) -> dict[str, dict[str, int]]:
    return {
        str(qid): {str(doc_id): int(rel) for doc_id, rel in docs.items()}
        for qid, docs in qrels.items()
    }


def _normalize_single_ranking(raw_ranking: Any) -> list[tuple[str, float]]:
    """Normalize a single query's ranking into an ordered ``[(doc_id, score)]`` list."""
    if raw_ranking is None:
        return []

    if isinstance(raw_ranking, Mapping):
        items = [
            (str(doc_id), float(score))
            for doc_id, score in raw_ranking.items()
        ]
        items.sort(key=lambda pair: pair[1], reverse=True)
        return items

    if isinstance(raw_ranking, Sequence) and not isinstance(raw_ranking, (str, bytes)):
        normalized: list[tuple[str, float]] = []
        for idx, item in enumerate(raw_ranking):
            if isinstance(item, Mapping):
                doc_id = item.get("doc_id") or item.get("_id") or item.get("id")
                if doc_id is None:
                    raise ValueError(f"Ranking entry at index {idx} missing doc_id.")
                score = float(item.get("score", 0.0))
                normalized.append((str(doc_id), score))
            elif isinstance(item, tuple) and len(item) >= 2:
                normalized.append((str(item[0]), float(item[1])))
            else:
                raise TypeError(
                    f"Unsupported ranking item at index {idx}: {type(item).__name__}"
                )
        return normalized

    raise TypeError(f"Unsupported ranking container type: {type(raw_ranking).__name__}")


def _normalize_rankings(
    rankings: Mapping[str, Any],
) -> dict[str, list[tuple[str, float]]]:
    return {
        str(qid): _normalize_single_ranking(raw)
        for qid, raw in rankings.items()
    }


def _deduplicate_preserve_order(
    ranked_docs: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    seen: set[str] = set()
    unique: list[tuple[str, float]] = []
    for doc_id, score in ranked_docs:
        if doc_id not in seen:
            seen.add(doc_id)
            unique.append((doc_id, score))
    return unique


class Evaluator:
    """Computes IR/RAG metrics, diagnostic comparisons, and required CSV exports."""

    @staticmethod
    def evaluate_query(
        query_qrels: Mapping[str, int],
        ranked_docs: Sequence[tuple[str, float]],
        k: int = 10,
    ) -> dict[str, float]:
        """Compute P@k, R@k, AP@k, Hit@k, and linear-gain NDCG@k for one query.

        Follows the exact conventions of Requirement 6 (and trec_eval /
        ir_measures):
        - Binary relevance (``rel > 0``) for Precision@k, Recall@k, AP@k, Hit@k.
        - Graded relevance preserved for NDCG@k with linear gain (``gain = rel``)
          and ideal ranking computed from all judgments in ``query_qrels``.
        - Denominator of Recall@k and AP@k is the total number of relevant
          documents in ``query_qrels``.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        clean_qrels = {str(d): int(r) for d, r in query_qrels.items()}
        total_relevant = sum(1 for r in clean_qrels.values() if r > 0)
        deduped = _deduplicate_preserve_order(list(ranked_docs))[:k]

        hits = 0
        ap_sum = 0.0
        dcg = 0.0

        for rank, (doc_id, _) in enumerate(deduped, start=1):
            rel = clean_qrels.get(doc_id, 0)
            if rel > 0:
                hits += 1
                ap_sum += hits / rank
                dcg += rel / math.log2(rank + 1)

        precision = hits / k
        hit_at_k = 1.0 if hits > 0 else 0.0

        if total_relevant > 0:
            recall = hits / total_relevant
            ap = ap_sum / total_relevant
        else:
            recall = 0.0
            ap = 0.0

        ideal_rels = sorted(
            (r for r in clean_qrels.values() if r > 0),
            reverse=True,
        )[:k]
        idcg = sum(
            rel / math.log2(rank + 1)
            for rank, rel in enumerate(ideal_rels, start=1)
        )
        ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

        return {
            f"P{k}": precision,
            f"R{k}": recall,
            f"AP{k}": ap,
            f"NDCG{k}": ndcg,
            f"Hit{k}": hit_at_k,
        }

    def evaluate(
        self,
        qrels: Mapping[str, Mapping[str, int | float]],
        rankings: Mapping[str, Any],
        k: int = 10,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate all queries present in ``qrels`` and return per-query and mean metrics."""
        norm_qrels = _normalize_qrels(qrels)
        norm_rankings = _normalize_rankings(rankings)

        if not norm_qrels:
            raise ValueError("qrels cannot be empty.")

        per_query: dict[str, dict[str, float]] = {}
        for qid in sorted(norm_qrels.keys()):
            ranked_docs = norm_rankings.get(qid, [])
            per_query[qid] = self.evaluate_query(norm_qrels[qid], ranked_docs, k=k)

        num_queries = len(per_query)
        mean_metrics = {
            f"Precision@{k}": sum(m[f"P{k}"] for m in per_query.values()) / num_queries,
            f"Recall@{k}": sum(m[f"R{k}"] for m in per_query.values()) / num_queries,
            f"MAP@{k}": sum(m[f"AP{k}"] for m in per_query.values()) / num_queries,
            f"NDCG@{k}": sum(m[f"NDCG{k}"] for m in per_query.values()) / num_queries,
            f"Hit@{k}": sum(m[f"Hit{k}"] for m in per_query.values()) / num_queries,
        }

        result: dict[str, Any] = {
            "num_queries": num_queries,
            "k": k,
            "global": mean_metrics,
            "per_query": per_query,
        }
        if model_name is not None:
            result["model"] = model_name
        return result

    @staticmethod
    def cross_check_with_pytrec_eval(
        qrels: Mapping[str, Mapping[str, int | float]],
        rankings: Mapping[str, Any],
    ) -> dict[str, dict[str, float]]:
        """Reference evaluation via ``pytrec_eval`` (P_10, recall_10, map_cut_10, ndcg_cut_10)."""
        norm_qrels = _normalize_qrels(qrels)
        norm_rankings = _normalize_rankings(rankings)
        run: dict[str, dict[str, float]] = {}
        for qid in norm_qrels:
            # Build strictly decreasing synthetic tie-breaker only if scores tie?
            # Here we pass actual scores directly as standard trec_eval input.
            docs = _deduplicate_preserve_order(norm_rankings.get(qid, []))
            run[qid] = {doc_id: float(score) for doc_id, score in docs}

        trec_evaluator = pytrec_eval.RelevanceEvaluator(
            norm_qrels,
            {"P_10", "recall_10", "map_cut_10", "ndcg_cut_10"},
        )
        return trec_evaluator.evaluate(run)

    def evaluate_rag_retrieval(
        self,
        qrels: Mapping[str, Mapping[str, int | float]],
        top5_rankings: Mapping[str, Any],
        query_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Compute Precision@5, Recall@5, and Hit@5 for the 10 RAG queries (Item 13)."""
        norm_qrels = _normalize_qrels(qrels)
        selected_qids = (
            [str(q) for q in query_ids]
            if query_ids is not None
            else sorted(norm_qrels.keys())
        )
        subset_qrels = {qid: norm_qrels[qid] for qid in selected_qids}
        return self.evaluate(subset_qrels, top5_rankings, k=5, model_name="RAG_Context")

    @staticmethod
    def select_rag_queries(
        test_qrels: Mapping[str, Any],
        queries: Mapping[str, str],
        seed: int = 42,
        sample_size: int = 10,
    ) -> list[str]:
        """Select the fixed RAG test queries ending with '?' using seed 42 (Item 12)."""
        ids = [
            q for q in sorted(test_qrels.keys())
            if str(queries[q]).strip().endswith("?")
        ]
        return random.Random(seed).sample(ids, sample_size)

    @staticmethod
    def compute_typo_recovery(
        ndcg_original: float,
        ndcg_error: float,
        ndcg_corrected: float,
    ) -> dict[str, Any]:
        """Compute typo recovery ratio per query (Requirement 4-(iv))."""
        denominator = ndcg_original - ndcg_error
        if denominator > 0:
            recovery = (ndcg_corrected - ndcg_error) / denominator
            return {
                "error_reduced_performance": True,
                "recovery_rate": recovery,
                "note": "",
            }
        return {
            "error_reduced_performance": False,
            "recovery_rate": None,
            "note": "o erro não reduziu o desempenho",
        }

    def evaluate_typo_experiment(
        self,
        qrels: Mapping[str, Mapping[str, int | float]],
        records: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Evaluate original, erroneous, and corrected rankings for the typo experiment."""
        norm_qrels = _normalize_qrels(qrels)
        evaluated: list[dict[str, Any]] = []

        for rec in records:
            qid = str(rec["query_id"])
            q_qrels = norm_qrels[qid]

            if "NDCG10_original" in rec:
                ndcg_orig = float(rec["NDCG10_original"])
            else:
                orig_rank = _normalize_single_ranking(rec["ranking_original"])
                ndcg_orig = self.evaluate_query(q_qrels, orig_rank, k=10)["NDCG10"]

            if ndcg_orig <= 0.0:
                raise ValueError(
                    f"Requirement 4-(iii) violated for query {qid}: "
                    f"NDCG@10 original ({ndcg_orig}) must be > 0."
                )

            if "NDCG10_erro" in rec:
                ndcg_err = float(rec["NDCG10_erro"])
            else:
                err_rank = _normalize_single_ranking(rec["ranking_erro"])
                ndcg_err = self.evaluate_query(q_qrels, err_rank, k=10)["NDCG10"]

            if "NDCG10_corrigida" in rec:
                ndcg_corr = float(rec["NDCG10_corrigida"])
            else:
                corr_rank = _normalize_single_ranking(rec["ranking_corrigida"])
                ndcg_corr = self.evaluate_query(q_qrels, corr_rank, k=10)["NDCG10"]

            recovery_info = self.compute_typo_recovery(ndcg_orig, ndcg_err, ndcg_corr)
            evaluated.append(
                {
                    "query_id": qid,
                    "query_original": str(rec["query_original"]),
                    "query_erro": str(rec["query_erro"]),
                    "query_corrigida": str(rec["query_corrigida"]),
                    "tipo_erro": str(rec["tipo_erro"]),
                    "NDCG10_original": ndcg_orig,
                    "NDCG10_erro": ndcg_err,
                    "NDCG10_corrigida": ndcg_corr,
                    **recovery_info,
                }
            )

        return evaluated

    def compare_lexical_vs_semantic(
        self,
        qrels: Mapping[str, Mapping[str, int | float]],
        lexical_rankings: Mapping[str, Any],
        semantic_rankings: Mapping[str, Any],
        top_n_queries: int = 3,
        top_docs_per_query: int = 5,
    ) -> dict[str, Any]:
        """Identify top-N semantic-advantage and lexical-advantage queries (Item 7)."""
        norm_qrels = _normalize_qrels(qrels)
        norm_lex = _normalize_rankings(lexical_rankings)
        norm_sem = _normalize_rankings(semantic_rankings)

        lex_eval = self.evaluate(norm_qrels, norm_lex, k=10, model_name="lexical")
        sem_eval = self.evaluate(norm_qrels, norm_sem, k=10, model_name="semantic")

        diffs: list[dict[str, Any]] = []
        for qid in sorted(norm_qrels.keys()):
            ndcg_lex = lex_eval["per_query"][qid]["NDCG10"]
            ndcg_sem = sem_eval["per_query"][qid]["NDCG10"]
            q_qrels = norm_qrels[qid]

            lex_top = [
                {
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": score,
                    "relevance": q_qrels.get(doc_id, 0),
                }
                for rank, (doc_id, score) in enumerate(
                    _deduplicate_preserve_order(norm_lex.get(qid, []))[:top_docs_per_query],
                    start=1,
                )
            ]
            sem_top = [
                {
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": score,
                    "relevance": q_qrels.get(doc_id, 0),
                }
                for rank, (doc_id, score) in enumerate(
                    _deduplicate_preserve_order(norm_sem.get(qid, []))[:top_docs_per_query],
                    start=1,
                )
            ]

            diffs.append(
                {
                    "query_id": qid,
                    "ndcg_lexical": ndcg_lex,
                    "ndcg_semantic": ndcg_sem,
                    "diff_semantic_minus_lexical": ndcg_sem - ndcg_lex,
                    "top_lexical": lex_top,
                    "top_semantic": sem_top,
                }
            )

        semantic_advantage = sorted(
            diffs,
            key=lambda x: (-x["diff_semantic_minus_lexical"], x["query_id"]),
        )[:top_n_queries]
        lexical_advantage = sorted(
            diffs,
            key=lambda x: (x["diff_semantic_minus_lexical"], x["query_id"]),
        )[:top_n_queries]

        return {
            "lexical_global": lex_eval["global"],
            "semantic_global": sem_eval["global"],
            "semantic_advantage": semantic_advantage,
            "lexical_advantage": lexical_advantage,
        }

    def analyze_reranking(
        self,
        qrels: Mapping[str, Mapping[str, int | float]],
        hybrid_rankings: Mapping[str, Any],
        reranked_rankings: Mapping[str, Any],
        top_gains: int = 2,
    ) -> dict[str, Any]:
        """Select largest gains, minimal change, and worst drop after reranking (Item 10)."""
        norm_qrels = _normalize_qrels(qrels)
        norm_hyb = _normalize_rankings(hybrid_rankings)
        norm_rer = _normalize_rankings(reranked_rankings)

        hyb_eval = self.evaluate(norm_qrels, norm_hyb, k=10, model_name="hybrid")
        rer_eval = self.evaluate(norm_qrels, norm_rer, k=10, model_name="reranking")

        cases: list[dict[str, Any]] = []
        for qid in sorted(norm_qrels.keys()):
            ndcg_before = hyb_eval["per_query"][qid]["NDCG10"]
            ndcg_after = rer_eval["per_query"][qid]["NDCG10"]
            q_qrels = norm_qrels[qid]

            hyb_list = _deduplicate_preserve_order(norm_hyb.get(qid, []))
            rer_list = _deduplicate_preserve_order(norm_rer.get(qid, []))

            pos_before = {doc_id: (rank, score) for rank, (doc_id, score) in enumerate(hyb_list, 1)}
            pos_after = {doc_id: (rank, score) for rank, (doc_id, score) in enumerate(rer_list, 1)}

            moved_docs: list[dict[str, Any]] = []
            for doc_id in set(pos_before) | set(pos_after):
                rb, sb = pos_before.get(doc_id, (None, None))
                ra, sa = pos_after.get(doc_id, (None, None))
                if rb != ra and ((rb is not None and rb <= 20) or (ra is not None and ra <= 20)):
                    delta = (rb - ra) if (rb is not None and ra is not None) else None
                    moved_docs.append(
                        {
                            "doc_id": doc_id,
                            "rank_before": rb,
                            "rank_after": ra,
                            "rank_gain": delta,
                            "score_before": sb,
                            "score_after": sa,
                            "relevance": q_qrels.get(doc_id, 0),
                        }
                    )
            moved_docs.sort(
                key=lambda d: abs(d["rank_gain"]) if d["rank_gain"] is not None else 0,
                reverse=True,
            )

            cases.append(
                {
                    "query_id": qid,
                    "ndcg_before": ndcg_before,
                    "ndcg_after": ndcg_after,
                    "ndcg_delta": ndcg_after - ndcg_before,
                    "moved_docs": moved_docs,
                }
            )

        sorted_by_gain = sorted(
            cases,
            key=lambda c: (-c["ndcg_delta"], c["query_id"]),
        )
        largest_gains = sorted_by_gain[:top_gains]
        chosen_ids = {c["query_id"] for c in largest_gains}

        remaining_for_drop = [c for c in cases if c["query_id"] not in chosen_ids]
        worst_drop = min(
            remaining_for_drop,
            key=lambda c: (c["ndcg_delta"], c["query_id"]),
        ) if remaining_for_drop else None
        if worst_drop is not None:
            chosen_ids.add(worst_drop["query_id"])

        remaining_for_unchanged = [c for c in cases if c["query_id"] not in chosen_ids]
        minimal_change = min(
            remaining_for_unchanged,
            key=lambda c: (abs(c["ndcg_delta"]), c["query_id"]),
        ) if remaining_for_unchanged else None

        return {
            "hybrid_global": hyb_eval["global"],
            "reranking_global": rer_eval["global"],
            "largest_gains": largest_gains,
            "minimal_change": minimal_change,
            "worst_drop": worst_drop,
        }

    @staticmethod
    def aggregate_rag_human_scores(
        rag_records: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, float | None]]:
        """Aggregate human scores (0-2) per condition, excluding 'N/A' from averages (Item 13)."""
        criteria = ("fidelidade", "completude", "utilidade", "rastreabilidade")
        summary: dict[str, dict[str, float | None]] = {}

        for condition in ("A", "B"):
            cond_rows = [
                r for r in rag_records
                if str(r.get("condition", "")).strip().upper() == condition
            ]
            cond_means: dict[str, float | None] = {}
            for criterion in criteria:
                valid_scores: list[float] = []
                for row in cond_rows:
                    val = row.get(criterion)
                    if val is None:
                        continue
                    if isinstance(val, str) and val.strip().upper() in ("", "N/A", "NA"):
                        continue
                    valid_scores.append(float(val))
                cond_means[criterion] = (
                    sum(valid_scores) / len(valid_scores)
                    if valid_scores
                    else None
                )
            summary[condition] = cond_means

        return summary

    @staticmethod
    def export_ranking_csv(
        rankings: Mapping[str, Any],
        qrels: Mapping[str, Mapping[str, int | float]],
        output_path: str | Path,
        top_k: int = 100,
    ) -> Path:
        """Export Top-100 ranking per query (Item 15: results_*.csv)."""
        norm_qrels = _normalize_qrels(qrels)
        norm_rankings = _normalize_rankings(rankings)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        query_ids = sorted(norm_qrels.keys()) if norm_qrels else sorted(norm_rankings.keys())
        fieldnames = ["query_id", "doc_id", "rank", "score", "relevance"]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for qid in query_ids:
                q_qrels = norm_qrels.get(qid, {})
                docs = _deduplicate_preserve_order(norm_rankings.get(qid, []))[:top_k]
                for rank, (doc_id, score) in enumerate(docs, start=1):
                    writer.writerow(
                        {
                            "query_id": qid,
                            "doc_id": doc_id,
                            "rank": rank,
                            "score": score,
                            "relevance": q_qrels.get(doc_id, 0),
                        }
                    )
        return path

    @staticmethod
    def export_metrics_per_query_csv(
        evaluations_by_model: Mapping[str, Mapping[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Export per-query metrics CSV (Item 15: metrics_per_query.csv)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["query_id", "model", "P10", "R10", "AP10", "NDCG10"]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for model_name, eval_result in evaluations_by_model.items():
                per_query = eval_result.get("per_query", eval_result)
                for qid in sorted(per_query.keys()):
                    m = per_query[qid]
                    writer.writerow(
                        {
                            "query_id": qid,
                            "model": model_name,
                            "P10": m["P10"],
                            "R10": m["R10"],
                            "AP10": m["AP10"],
                            "NDCG10": m["NDCG10"],
                        }
                    )
        return path

    @staticmethod
    def export_typo_experiment_csv(
        typo_records: Sequence[Mapping[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Export typo experiment results CSV (Item 15: typo_experiment.csv)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "query_id",
            "query_original",
            "query_erro",
            "query_corrigida",
            "tipo_erro",
            "NDCG10_original",
            "NDCG10_erro",
            "NDCG10_corrigida",
            "recovery_rate",
            "error_reduced_performance",
            "note",
        ]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in typo_records:
                writer.writerow({field: rec.get(field, "") for field in fieldnames})
        return path

    @staticmethod
    def export_rag_experiment_csv(
        rag_records: Sequence[Mapping[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Export RAG experiment A/B CSV (Item 15: rag_experiment.csv)."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "query_id",
            "query",
            "condition",
            "context_doc_ids",
            "prompt",
            "answer",
            "fidelidade",
            "justificativa_fidelidade",
            "completude",
            "justificativa_completude",
            "utilidade",
            "justificativa_utilidade",
            "rastreabilidade",
            "justificativa_rastreabilidade",
        ]

        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in rag_records:
                row = dict(rec)
                cond = str(row.get("condition", "")).strip().upper()
                doc_ids = row.get("context_doc_ids", [])
                if isinstance(doc_ids, (list, tuple)):
                    row["context_doc_ids"] = ";".join(str(d) for d in doc_ids)
                if cond == "A" and row.get("rastreabilidade") in (None, ""):
                    row["rastreabilidade"] = "N/A"
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        return path
