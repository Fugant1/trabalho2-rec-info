"""Diagnostic analysis and human-evaluation workflow helpers for Trabalho Pratico 2.

Provides structured, data-backed evidence extraction for:
- Requirement 7: Lexical vs. Semantic per-query comparison (term matches, TF/DF/IDF,
  doc lengths, cross-system ranks, snippets).
- Requirement 10: Reranking movement analysis (largest gains, minimal change,
  worst drop with Top-10 before/after and document details).
- Requirement 13: Blinded A/B human evaluation sheet generation, unblinding/merging
  back into ``rag_experiment.csv``, condition averages, and inter-annotator
  agreement (exact agreement + Cohen's Kappa).
- Requirement 14: End-to-end 6-stage RAG funnel trace (lexical -> semantic ->
  hybrid -> reranking -> context selection -> generation) to localize failures.
"""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluator import Evaluator

EXPECTED_NFCORPUS_RAG_QUERIES: tuple[str, ...] = (
    "PLAIN-68",
    "PLAIN-2450",
    "PLAIN-153",
    "PLAIN-2850",
    "PLAIN-2750",
    "PLAIN-270",
    "PLAIN-2470",
    "PLAIN-2430",
    "PLAIN-238",
    "PLAIN-3342",
)

_RAG_CRITERIA: tuple[str, ...] = (
    "fidelidade",
    "completude",
    "utilidade",
    "rastreabilidade",
)


def _doc_title_and_snippet(
    corpus: Mapping[str, Any],
    doc_id: str,
    snippet_chars: int = 320,
) -> tuple[str, str]:
    raw = corpus.get(doc_id)
    if raw is None:
        return "", ""
    if isinstance(raw, Mapping):
        title = str(raw.get("title") or "").strip()
        text = str(raw.get("text") or "").strip()
    else:
        title = ""
        text = str(raw).strip()
    snippet = text[:snippet_chars] + ("..." if len(text) > snippet_chars else "")
    return title, snippet


def _rank_lookup(ranking: Sequence[tuple[str, float]]) -> dict[str, tuple[int, float]]:
    return {
        str(doc_id): (rank, float(score))
        for rank, (doc_id, score) in enumerate(ranking, start=1)
    }


def enrich_lexical_vs_semantic_analysis(
    comparison: Mapping[str, Any],
    queries: Mapping[str, str],
    corpus: Mapping[str, Any],
    lexical_retriever: Any,
    lexical_rankings: Mapping[str, Sequence[tuple[str, float]]],
    semantic_rankings: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int | float]],
    snippet_chars: int = 320,
) -> dict[str, Any]:
    """Attach concrete corpus/index evidence to the 6 Req-7 advantage queries."""

    def _enrich_case(case: Mapping[str, Any]) -> dict[str, Any]:
        qid = str(case["query_id"])
        qtext = str(queries.get(qid, ""))
        q_tokens = lexical_retriever.tokenize_query(qtext)
        term_stats = [
            {
                "term": term,
                "df": lexical_retriever.df.get(term, 0),
                "idf": lexical_retriever.idf.get(term, 0.0),
            }
            for term in dict.fromkeys(q_tokens)
        ]

        lex_pos = _rank_lookup(lexical_rankings.get(qid, []))
        sem_pos = _rank_lookup(semantic_rankings.get(qid, []))
        q_qrels = qrels.get(qid, {})
        total_rel_docs = sum(1 for r in q_qrels.values() if int(r) > 0)

        def _enrich_top_list(
            top_entries: Sequence[Mapping[str, Any]],
            other_lookup: Mapping[str, tuple[int, float]],
        ) -> list[dict[str, Any]]:
            enriched: list[dict[str, Any]] = []
            for entry in top_entries:
                doc_id = str(entry["doc_id"])
                title, snippet = _doc_title_and_snippet(corpus, doc_id, snippet_chars)
                lex_evidence = lexical_retriever.explain_doc_match(qtext, doc_id)
                other_rank, other_score = other_lookup.get(doc_id, (None, None))
                enriched.append(
                    {
                        **dict(entry),
                        "title": title,
                        "text_snippet": snippet,
                        "doc_length_tokens": lex_evidence["doc_length_tokens"],
                        "avg_doc_length_tokens": lex_evidence["avg_doc_length_tokens"],
                        "lexical_overlap_ratio": lex_evidence["lexical_overlap_ratio"],
                        "matched_query_terms": lex_evidence["matched_terms"],
                        "missing_query_terms": lex_evidence["missing_terms"],
                        "rank_in_other_system": other_rank,
                        "score_in_other_system": other_score,
                    }
                )
            return enriched

        return {
            "query_id": qid,
            "query_text": qtext,
            "query_tokens": q_tokens,
            "query_term_stats": term_stats,
            "total_relevant_docs_in_qrels": total_rel_docs,
            "ndcg_lexical": case["ndcg_lexical"],
            "ndcg_semantic": case["ndcg_semantic"],
            "diff_semantic_minus_lexical": case["diff_semantic_minus_lexical"],
            "top_lexical": _enrich_top_list(case["top_lexical"], sem_pos),
            "top_semantic": _enrich_top_list(case["top_semantic"], lex_pos),
        }

    return {
        "lexical_stemming": getattr(lexical_retriever, "stemming", "none"),
        "lexical_global": comparison["lexical_global"],
        "semantic_global": comparison["semantic_global"],
        "semantic_advantage": [
            _enrich_case(c) for c in comparison["semantic_advantage"]
        ],
        "lexical_advantage": [
            _enrich_case(c) for c in comparison["lexical_advantage"]
        ],
    }


def enrich_reranking_analysis(
    reranking_analysis: Mapping[str, Any],
    queries: Mapping[str, str],
    corpus: Mapping[str, Any],
    hybrid_rankings: Mapping[str, Sequence[tuple[str, float]]],
    reranked_rankings: Mapping[str, Sequence[tuple[str, float]]],
    qrels: Mapping[str, Mapping[str, int | float]],
    top_k: int = 10,
    snippet_chars: int = 280,
) -> dict[str, Any]:
    """Attach query texts, Top-10 before/after, and snippets to Req-10 cases."""

    def _enrich_case(case: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if case is None:
            return None
        qid = str(case["query_id"])
        qtext = str(queries.get(qid, ""))
        q_qrels = qrels.get(qid, {})

        hyb_list = list(hybrid_rankings.get(qid, []))
        rer_list = list(reranked_rankings.get(qid, []))
        hyb_lookup = _rank_lookup(hyb_list)
        rer_lookup = _rank_lookup(rer_list)

        top_before = []
        for rank, (doc_id, score) in enumerate(hyb_list[:top_k], start=1):
            title, snippet = _doc_title_and_snippet(corpus, doc_id, snippet_chars)
            after_rank, after_score = rer_lookup.get(doc_id, (None, None))
            top_before.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text_snippet": snippet,
                    "rank_before": rank,
                    "score_before": float(score),
                    "rank_after": after_rank,
                    "score_after": after_score,
                    "rank_gain": (rank - after_rank) if after_rank is not None else None,
                    "relevance": int(q_qrels.get(doc_id, 0)),
                }
            )

        top_after = []
        for rank, (doc_id, score) in enumerate(rer_list[:top_k], start=1):
            title, snippet = _doc_title_and_snippet(corpus, doc_id, snippet_chars)
            before_rank, before_score = hyb_lookup.get(doc_id, (None, None))
            top_after.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text_snippet": snippet,
                    "rank_after": rank,
                    "score_after": float(score),
                    "rank_before": before_rank,
                    "score_before": before_score,
                    "rank_gain": (before_rank - rank) if before_rank is not None else None,
                    "relevance": int(q_qrels.get(doc_id, 0)),
                }
            )

        moved_enriched = []
        for doc in case.get("moved_docs", [])[:15]:
            doc_id = str(doc["doc_id"])
            title, snippet = _doc_title_and_snippet(corpus, doc_id, snippet_chars)
            moved_enriched.append(
                {
                    **dict(doc),
                    "title": title,
                    "text_snippet": snippet,
                }
            )

        return {
            "query_id": qid,
            "query_text": qtext,
            "ndcg_before": case["ndcg_before"],
            "ndcg_after": case["ndcg_after"],
            "ndcg_delta": case["ndcg_delta"],
            "top_before": top_before,
            "top_after": top_after,
            "moved_docs": moved_enriched,
        }

    return {
        "hybrid_global": reranking_analysis["hybrid_global"],
        "reranking_global": reranking_analysis["reranking_global"],
        "largest_gains": [
            _enrich_case(c) for c in reranking_analysis.get("largest_gains", [])
        ],
        "minimal_change": _enrich_case(reranking_analysis.get("minimal_change")),
        "worst_drop": _enrich_case(reranking_analysis.get("worst_drop")),
    }


def build_rag_blind_evaluation(
    rag_records: Sequence[Mapping[str, Any]],
    blind_csv_path: str | Path,
    key_json_path: str | Path,
    *,
    corpus: Mapping[str, Any] | None = None,
    context_reference_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a blinded A/B evaluation CSV shuffling Condition A and B per query.

    Follows Requirement 13's recommendation: annotators score answers without
    knowing whether each answer came from Condition A (no RAG) or Condition B
    (with RAG). A separate ``key_json_path`` stores the mapping for unblinding,
    and an optional ``context_reference_path`` stores the Top-5 context texts
    per query so annotators can evaluate Fidelity consistently across both
    conditions.
    """
    rng = random.Random(seed)

    by_query: dict[str, list[dict[str, Any]]] = {}
    for rec in rag_records:
        qid = str(rec["query_id"])
        by_query.setdefault(qid, []).append(dict(rec))

    blind_rows: list[dict[str, Any]] = []
    blind_key: dict[str, dict[str, Any]] = {}
    context_ref: dict[str, Any] = {}

    for qid in sorted(by_query.keys()):
        pair = list(by_query[qid])
        # Find shared Top-5 context docs (from Condition B) for Fidelity evaluation
        cond_b = next(
            (r for r in pair if str(r.get("condition", "")).strip().upper() == "B"),
            None,
        )
        shared_doc_ids: list[str] = []
        if cond_b is not None:
            raw_ids = cond_b.get("context_doc_ids", [])
            if isinstance(raw_ids, str):
                shared_doc_ids = [x.strip() for x in raw_ids.split(";") if x.strip()]
            else:
                shared_doc_ids = [str(x) for x in raw_ids]

        if corpus is not None:
            context_ref[qid] = {
                "query": pair[0].get("query", ""),
                "top5_context_docs": [
                    {
                        "doc_id": doc_id,
                        "title": (
                            str(corpus[doc_id].get("title", "")).strip()
                            if isinstance(corpus.get(doc_id), Mapping)
                            else ""
                        ),
                        "text": (
                            str(corpus[doc_id].get("text", "")).strip()
                            if isinstance(corpus.get(doc_id), Mapping)
                            else str(corpus.get(doc_id, ""))
                        ),
                    }
                    for doc_id in shared_doc_ids
                ],
            }

        rng.shuffle(pair)
        for slot_idx, rec in enumerate(pair, start=1):
            blind_id = f"{qid}__R{slot_idx}"
            cond = str(rec.get("condition", "")).strip().upper()
            blind_key[blind_id] = {
                "query_id": qid,
                "condition": cond,
                "slot": slot_idx,
            }
            blind_rows.append(
                {
                    "blind_id": blind_id,
                    "query_id": qid,
                    "query": rec.get("query", ""),
                    "reference_top5_doc_ids": ";".join(shared_doc_ids),
                    "answer": rec.get("answer", ""),
                    "fidelidade": "",
                    "justificativa_fidelidade": "",
                    "completude": "",
                    "justificativa_completude": "",
                    "utilidade": "",
                    "justificativa_utilidade": "",
                    "rastreabilidade": "",
                    "justificativa_rastreabilidade": "",
                }
            )

    blind_path = Path(blind_csv_path)
    blind_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "blind_id",
        "query_id",
        "query",
        "reference_top5_doc_ids",
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
    with open(blind_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(blind_rows)

    key_path = Path(key_json_path)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": seed, "mapping": blind_key},
            f,
            indent=2,
            ensure_ascii=False,
        )

    if context_reference_path is not None and context_ref:
        ref_path = Path(context_reference_path)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(context_ref, f, indent=2, ensure_ascii=False)

    return {
        "blind_csv": str(blind_path),
        "key_json": str(key_path),
        "context_reference_json": str(context_reference_path)
        if context_reference_path is not None
        else None,
        "num_rows": len(blind_rows),
    }


def compute_cohens_kappa(
    scores_a: Sequence[int],
    scores_b: Sequence[int],
) -> float | None:
    """Compute unweighted Cohen's Kappa between two discrete annotator sequences."""
    if len(scores_a) != len(scores_b) or len(scores_a) == 0:
        return None

    n = len(scores_a)
    observed_agree = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n

    counts_a = Counter(scores_a)
    counts_b = Counter(scores_b)
    labels = set(counts_a.keys()) | set(counts_b.keys())
    expected_agree = sum((counts_a[lbl] / n) * (counts_b[lbl] / n) for lbl in labels)

    if abs(1.0 - expected_agree) < 1e-12:
        return 1.0 if abs(1.0 - observed_agree) < 1e-12 else 0.0
    return (observed_agree - expected_agree) / (1.0 - expected_agree)


def _parse_optional_score(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.upper() in ("N/A", "NA"):
        return None
    val = float(s)
    if val not in (0.0, 1.0, 2.0):
        raise ValueError(f"Human score must be 0, 1, 2, or N/A; got '{raw}'.")
    return val


def merge_blind_evaluations(
    rag_csv_path: str | Path,
    blind_csv_paths: Sequence[str | Path],
    key_json_path: str | Path,
    output_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Unblind 1 or 2 annotators' sheets, update ``rag_experiment.csv``, and aggregate.

    When 2 annotator CSVs are supplied:
    - Computes exact agreement and Cohen's Kappa per criterion.
    - Averages the two annotators' numeric scores in the merged output and
      concatenates both justifications.
    - Enforces ``rastreabilidade = "N/A"`` for Condition A per Requirement 13.
    """
    if not blind_csv_paths:
        raise ValueError("At least one completed blind evaluation CSV is required.")

    with open(key_json_path, "r", encoding="utf-8") as f:
        key_payload = json.load(f)
    blind_key: dict[str, dict[str, Any]] = key_payload["mapping"]

    with open(rag_csv_path, "r", encoding="utf-8", newline="") as f:
        base_rows = list(csv.DictReader(f))

    # Load each annotator's rows keyed by (query_id, condition)
    annotator_maps: list[dict[tuple[str, str], dict[str, str]]] = []
    for path in blind_csv_paths:
        amap: dict[tuple[str, str], dict[str, str]] = {}
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                bid = row["blind_id"].strip()
                if bid not in blind_key:
                    raise KeyError(f"Unknown blind_id '{bid}' in {path}.")
                meta = blind_key[bid]
                amap[(meta["query_id"], meta["condition"])] = row
        annotator_maps.append(amap)

    merged_rows: list[dict[str, Any]] = []
    for base in base_rows:
        qid = base["query_id"].strip()
        cond = base["condition"].strip().upper()
        key = (qid, cond)

        updated = dict(base)
        for crit in _RAG_CRITERIA:
            just_col = f"justificativa_{crit}"
            if cond == "A" and crit == "rastreabilidade":
                updated[crit] = "N/A"
                justs = [
                    amap[key].get(just_col, "").strip()
                    for amap in annotator_maps
                    if key in amap and amap[key].get(just_col, "").strip()
                ]
                updated[just_col] = (
                    " | ".join(justs)
                    if justs
                    else "Condicao sem RAG: nenhuma fonte externa fornecida."
                )
                continue

            parsed_vals: list[float] = []
            justs = []
            for idx, amap in enumerate(annotator_maps, start=1):
                if key not in amap:
                    continue
                val = _parse_optional_score(amap[key].get(crit))
                if val is not None:
                    parsed_vals.append(val)
                jtxt = amap[key].get(just_col, "").strip()
                if jtxt:
                    prefix = f"[Avaliador {idx}] " if len(annotator_maps) > 1 else ""
                    justs.append(f"{prefix}{jtxt}")

            if parsed_vals:
                avg_val = sum(parsed_vals) / len(parsed_vals)
                updated[crit] = int(avg_val) if avg_val.is_integer() else round(avg_val, 3)
            updated[just_col] = " | ".join(justs)

        merged_rows.append(updated)

    # Inter-annotator agreement when 2 annotators are provided
    agreement: dict[str, Any] | None = None
    if len(annotator_maps) >= 2:
        map1, map2 = annotator_maps[0], annotator_maps[1]
        shared_keys = sorted(set(map1.keys()) & set(map2.keys()))
        agreement = {}
        for crit in _RAG_CRITERIA:
            s1: list[int] = []
            s2: list[int] = []
            for qid, cond in shared_keys:
                if cond == "A" and crit == "rastreabilidade":
                    continue
                v1 = _parse_optional_score(map1[(qid, cond)].get(crit))
                v2 = _parse_optional_score(map2[(qid, cond)].get(crit))
                if v1 is not None and v2 is not None:
                    s1.append(int(v1))
                    s2.append(int(v2))
            exact = (
                sum(1 for a, b in zip(s1, s2) if a == b) / len(s1)
                if s1
                else None
            )
            agreement[crit] = {
                "n_pairs": len(s1),
                "exact_agreement": exact,
                "cohens_kappa": compute_cohens_kappa(s1, s2),
            }

    target_path = Path(output_csv_path) if output_csv_path is not None else Path(rag_csv_path)
    Evaluator.export_rag_experiment_csv(merged_rows, target_path)

    condition_means = Evaluator.aggregate_rag_human_scores(merged_rows)
    return {
        "output_csv": str(target_path),
        "condition_means": condition_means,
        "inter_annotator_agreement": agreement,
    }


def _extract_cited_doc_ids(answer: str) -> list[str]:
    """Extract cited document IDs such as ``[MED-123]`` or ``MED-123`` from an answer."""
    # Match common BEIR/NFCorpus doc IDs (e.g., MED-1234, PLAIN-12) or bracketed IDs
    bracketed = re.findall(r"\[([A-Za-z0-9_-]+)\]", answer)
    prefixed = re.findall(r"\b((?:MED|PLAIN|DOC)-\d+)\b", answer)
    combined: list[str] = []
    for item in bracketed + prefixed:
        if item not in combined:
            combined.append(item)
    return combined


def build_end_to_end_rag_trace(
    rag_query_ids: Sequence[str],
    queries: Mapping[str, str],
    corpus: Mapping[str, Any],
    qrels: Mapping[str, Mapping[str, int | float]],
    lexical_rankings: Mapping[str, Sequence[tuple[str, float]]],
    semantic_rankings: Mapping[str, Sequence[tuple[str, float]]],
    hybrid_rankings: Mapping[str, Sequence[tuple[str, float]]],
    reranked_rankings: Mapping[str, Sequence[tuple[str, float]]],
    rag_records: Sequence[Mapping[str, Any]],
    top_k_context: int = 5,
) -> dict[str, Any]:
    """Build a 6-stage pipeline trace for each RAG query to localize failures (Req 14)."""
    evaluator = Evaluator()

    records_by_qc: dict[tuple[str, str], Mapping[str, Any]] = {
        (str(r["query_id"]), str(r["condition"]).strip().upper()): r
        for r in rag_records
    }

    traces: dict[str, Any] = {}
    for qid in rag_query_ids:
        qid_str = str(qid)
        qtext = str(queries.get(qid_str, ""))
        q_qrels = {str(d): int(rel) for d, rel in qrels.get(qid_str, {}).items()}
        relevant_docs = {d for d, rel in q_qrels.items() if rel > 0}

        lex = list(lexical_rankings.get(qid_str, []))
        sem = list(semantic_rankings.get(qid_str, []))
        hyb = list(hybrid_rankings.get(qid_str, []))
        rer = list(reranked_rankings.get(qid_str, []))

        lex_top100_ids = {d for d, _ in lex[:100]}
        sem_top100_ids = {d for d, _ in sem[:100]}
        hyb_top50_ids = {d for d, _ in hyb[:50]}
        hyb_top100_ids = {d for d, _ in hyb[:100]}
        rer_top5_ids = [d for d, _ in rer[:top_k_context]]

        lex_eval = evaluator.evaluate_query(q_qrels, lex, k=10)
        sem_eval = evaluator.evaluate_query(q_qrels, sem, k=10)
        hyb_eval = evaluator.evaluate_query(q_qrels, hyb, k=10)
        rer_eval = evaluator.evaluate_query(q_qrels, rer, k=10)
        ctx_eval = evaluator.evaluate_query(q_qrels, rer[:top_k_context], k=top_k_context)

        lex_pos = _rank_lookup(lex)
        sem_pos = _rank_lookup(sem)
        hyb_pos = _rank_lookup(hyb)
        rer_pos = _rank_lookup(rer)

        context_docs_detail = []
        for rank, doc_id in enumerate(rer_top5_ids, start=1):
            title, snippet = _doc_title_and_snippet(corpus, doc_id, snippet_chars=260)
            context_docs_detail.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text_snippet": snippet,
                    "relevance": q_qrels.get(doc_id, 0),
                    "rank_lexical": lex_pos.get(doc_id, (None, None))[0],
                    "rank_semantic": sem_pos.get(doc_id, (None, None))[0],
                    "rank_hybrid": hyb_pos.get(doc_id, (None, None))[0],
                    "rank_reranking": rank,
                    "score_reranking": rer_pos.get(doc_id, (None, None))[1],
                }
            )

        rec_a = records_by_qc.get((qid_str, "A"), {})
        rec_b = records_by_qc.get((qid_str, "B"), {})
        answer_b = str(rec_b.get("answer", ""))
        cited_ids = _extract_cited_doc_ids(answer_b)
        ctx_id_set = set(rer_top5_ids)

        traces[qid_str] = {
            "query_id": qid_str,
            "query": qtext,
            "total_relevant_docs_in_qrels": len(relevant_docs),
            "stage_1_lexical": {
                "NDCG10": lex_eval["NDCG10"],
                "P10": lex_eval["P10"],
                "R10": lex_eval["R10"],
                "relevant_in_top100": len(relevant_docs & lex_top100_ids),
            },
            "stage_2_semantic": {
                "NDCG10": sem_eval["NDCG10"],
                "P10": sem_eval["P10"],
                "R10": sem_eval["R10"],
                "relevant_in_top100": len(relevant_docs & sem_top100_ids),
            },
            "stage_3_hybrid": {
                "NDCG10": hyb_eval["NDCG10"],
                "P10": hyb_eval["P10"],
                "R10": hyb_eval["R10"],
                "relevant_in_top100": len(relevant_docs & hyb_top100_ids),
                "relevant_in_top50": len(relevant_docs & hyb_top50_ids),
                "relevant_available_from_union_top100": len(
                    relevant_docs & (lex_top100_ids | sem_top100_ids)
                ),
            },
            "stage_4_reranking": {
                "NDCG10": rer_eval["NDCG10"],
                "P10": rer_eval["P10"],
                "R10": rer_eval["R10"],
                "ndcg10_delta_vs_hybrid": rer_eval["NDCG10"] - hyb_eval["NDCG10"],
            },
            "stage_5_context_selection": {
                f"Precision@{top_k_context}": ctx_eval[f"P{top_k_context}"],
                f"Recall@{top_k_context}": ctx_eval[f"R{top_k_context}"],
                f"Hit@{top_k_context}": ctx_eval[f"Hit{top_k_context}"],
                "relevant_in_context": len(relevant_docs & ctx_id_set),
                "context_docs": context_docs_detail,
            },
            "stage_6_generation": {
                "condition_A_answer": str(rec_a.get("answer", "")),
                "condition_B_answer": answer_b,
                "condition_B_cited_ids": cited_ids,
                "condition_B_valid_citations": [c for c in cited_ids if c in ctx_id_set],
                "condition_B_hallucinated_citations": [
                    c for c in cited_ids if c not in ctx_id_set
                ],
            },
        }

    return traces
