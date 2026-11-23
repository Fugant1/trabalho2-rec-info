"""End-to-end experiment orchestrator for Trabalho Pratico 2 (SCC0282 - IR).

Coordinates every stage specified in ``Trabalho_Pratico_2_RI_2026.pdf``:
1. Dataset loading (BEIR NFCorpus train/dev/test) and preprocessing/indexing.
2. Dev-set hyperparameter tuning (RRF ``k in {10, 60, 100}``, optional BM25
   ``(k1, b)`` and SpellCorrector tuning).
3. Test-set retrieval & evaluation (Lexical BM25, Semantic Bi-Encoder, Hybrid
   RRF, and Cross-Encoder Reranking of Top-50), standardized CSV exports, and
   enriched Req 7 / Req 10 diagnostic reports.
4. Separate spelling-correction (typo tolerance) experiment on 10 eligible test
   queries (Req 4) + ``typo_experiment.csv``.
5. RAG experiment (Condition A without RAG vs. Condition B with RAG on the 10
   fixed seed-42 question queries, Reqs 11-14), retrieval metrics at @5,
   ``rag_experiment.csv``, blinded human-evaluation sheets, and 6-stage
   end-to-end failure trace.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation import (
    EXPECTED_NFCORPUS_RAG_QUERIES,
    Evaluator,
    build_end_to_end_rag_trace,
    build_rag_blind_evaluation,
    enrich_lexical_vs_semantic_analysis,
    enrich_reranking_analysis,
    merge_blind_evaluations,
)
from src.pre_processing import (
    DEFAULT_EMBEDDING_MODEL,
    PreProcessor,
    SpellCorrector,
    build_typo_benchmark,
    tune_spell_corrector_on_dev,
)
from src.retrieval import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_RERANKER_MODEL,
    BM25Retriever,
    CrossEncoderReranker,
    HybridRetriever,
    RAGLLM,
    Retriever,
    SemanticRetriever,
)
from src.utils import DataLoader


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        val = config.get(key, default)
        return default if val is None else val
    val = getattr(config, key, default)
    return default if val is None else val


def _collect_environment_info() -> dict[str, Any]:
    """Collect Python, library versions, and basic CPU/GPU hardware info."""
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "sentence-transformers",
        "transformers",
        "beir",
        "pytrec_eval",
        "langchain-core",
        "langchain-google-genai",
    )
    pkg_versions: dict[str, str | None] = {}
    try:
        from importlib import metadata as importlib_metadata

        for pkg in packages:
            try:
                pkg_versions[pkg] = importlib_metadata.version(pkg)
            except importlib_metadata.PackageNotFoundError:
                pkg_versions[pkg] = None
    except Exception:
        pass

    gpu_info: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info = {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass

    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": platform.os.cpu_count(),
        "gpu": gpu_info,
        "packages": pkg_versions,
    }


def _load_ranking_csv(csv_path: Path) -> dict[str, list[tuple[str, float]]]:
    """Reload a standardized ``results_*.csv`` file back into a rankings dict."""
    grouped: dict[str, list[tuple[int, str, float]]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            qid = str(row["query_id"])
            rank = int(row["rank"])
            doc_id = str(row["doc_id"])
            score = float(row["score"])
            grouped.setdefault(qid, []).append((rank, doc_id, score))

    rankings: dict[str, list[tuple[str, float]]] = {}
    for qid, entries in grouped.items():
        entries.sort(key=lambda item: item[0])
        rankings[qid] = [(doc_id, score) for _, doc_id, score in entries]
    return rankings


class Experiment:
    """Orchestrates all experimental stages for Trabalho Pratico 2."""

    def __init__(
        self,
        config: Any = None,
        *,
        data_loader: Any | None = None,
        embedding_model: Any | None = None,
        reranker_model: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.config = config if config is not None else {}

        self.dataset_name: str = str(_cfg_get(self.config, "dataset_name", "nfcorpus"))
        self.data_dir = Path(_cfg_get(self.config, "data_dir", "datasets"))
        self.cache_dir = Path(_cfg_get(self.config, "cache_dir", "cache"))
        self.runs_dir = Path(_cfg_get(self.config, "runs_dir", "runs"))
        self.results_dir = Path(_cfg_get(self.config, "results_dir", "results"))
        self.dev_results_dir = self.results_dir / "ajuste_dev"
        self.retrieval_results_dir = self.results_dir / "recuperacao"
        self.typo_results_dir = self.results_dir / "tolerancia_erros"
        self.rag_results_dir = self.results_dir / "rag"
        self.human_evaluation_dir = self.rag_results_dir / "avaliacao_humana"
        self.run_id: str = str(
            _cfg_get(
                self.config,
                "run_id",
                datetime.now().strftime("run_%Y%m%d_%H%M%S"),
            )
        )
        self.run_dir = self.runs_dir / self.run_id

        # Model and experiment hyperparameters
        self.bm25_k1: float = float(_cfg_get(self.config, "bm25_k1", 1.5))
        self.bm25_b: float = float(_cfg_get(self.config, "bm25_b", 0.75))
        self.tune_bm25_on_dev: bool = bool(
            _cfg_get(self.config, "tune_bm25_on_dev", False)
        )

        self.embedding_model_name: str = str(
            _cfg_get(self.config, "embedding_model_name", DEFAULT_EMBEDDING_MODEL)
        )
        self.embedding_device = _cfg_get(self.config, "embedding_device", None)
        self.semantic_batch_size = _cfg_get(self.config, "semantic_batch_size", None)

        self.rrf_k: int = int(_cfg_get(self.config, "rrf_k", 60))
        self.rrf_k_candidates: list[int] = [
            int(k)
            for k in _cfg_get(self.config, "rrf_k_candidates", [10, 60, 100])
        ]
        self.rrf_fusion_depth: int = int(
            _cfg_get(self.config, "rrf_fusion_depth", 100)
        )

        self.reranker_model_name: str = str(
            _cfg_get(self.config, "reranker_model_name", DEFAULT_RERANKER_MODEL)
        )
        self.rerank_top_k: int = int(_cfg_get(self.config, "rerank_top_k", 50))
        self.rerank_batch_size: int = int(
            _cfg_get(self.config, "rerank_batch_size", 32)
        )
        self.reranker_device = _cfg_get(self.config, "reranker_device", None)

        # Typo experiment configuration (Requirement 4)
        self.typo_k_gram: int = int(_cfg_get(self.config, "typo_k_gram", 2))
        self.typo_jaccard_threshold: float = float(
            _cfg_get(self.config, "typo_jaccard_threshold", 0.25)
        )
        self.typo_max_edit_distance: int = int(
            _cfg_get(self.config, "typo_max_edit_distance", 2)
        )
        self.typo_distance_metric: str = str(
            _cfg_get(self.config, "typo_distance_metric", "damerau_levenshtein")
        )
        self.typo_tune_on_dev: bool = bool(
            _cfg_get(self.config, "typo_tune_on_dev", True)
        )
        self.typo_num_queries: int = int(_cfg_get(self.config, "typo_num_queries", 10))
        self.typo_seed: int = int(_cfg_get(self.config, "typo_seed", 42))
        self.typo_eval_semantic: bool = bool(
            _cfg_get(self.config, "typo_eval_semantic", True)
        )
        self.typo_queries_path = _cfg_get(self.config, "typo_queries_path", None)

        # RAG experiment configuration (Requirements 11-14)
        self.rag_seed: int = int(_cfg_get(self.config, "rag_seed", 42))
        self.rag_sample_size: int = int(_cfg_get(self.config, "rag_sample_size", 10))
        self.rag_top_k: int = int(_cfg_get(self.config, "rag_top_k", 5))
        self.llm_model: str = str(
            _cfg_get(self.config, "llm_model", DEFAULT_QWEN_MODEL)
        )
        self.llm_provider: str = str(
            _cfg_get(self.config, "llm_provider", "auto")
        )
        self.llm_load_in_4bit: bool = bool(
            _cfg_get(self.config, "llm_load_in_4bit", True)
        )
        self.llm_enable_thinking: bool = bool(
            _cfg_get(self.config, "llm_enable_thinking", False)
        )
        self.llm_device_map: str = str(
            _cfg_get(self.config, "llm_device_map", "auto")
        )
        self.llm_temperature: float = float(
            _cfg_get(self.config, "llm_temperature", 0.0)
        )
        self.llm_max_output_tokens: int | None = _cfg_get(
            self.config, "llm_max_output_tokens", 1024
        )
        self.llm_seed: int | None = _cfg_get(self.config, "llm_seed", 42)
        self.llm_language: str = str(
            _cfg_get(self.config, "llm_language", "Portuguese")
        )
        self.google_api_key = _cfg_get(self.config, "google_api_key", None)

        # Injectable components (facilitates unit testing & customization)
        self.data_loader = data_loader or DataLoader(
            dataset_name=self.dataset_name,
            data_dir=self.data_dir,
        )
        self._injected_embedding_model = embedding_model
        self._injected_reranker_model = reranker_model
        self._injected_llm_client = llm_client

        preprocessor_cfg = {
            "cache_dir": str(self.cache_dir),
            "lexical_cache_file": _cfg_get(self.config, "lexical_cache_file"),
            "lexical_stemming": _cfg_get(self.config, "lexical_stemming", "none"),
            "runs_dir": str(self.runs_dir),
            "run_dir": str(self.run_dir),
            "force_rebuild": bool(_cfg_get(self.config, "force_rebuild", False)),
            "embedding_model_name": self.embedding_model_name,
            "embedding_device": self.embedding_device,
            "embedding_batch_size": int(
                _cfg_get(self.config, "embedding_batch_size", 64)
            ),
            "normalize_embeddings": bool(
                _cfg_get(self.config, "normalize_embeddings", True)
            ),
            "embedding_model": self._injected_embedding_model,
            "lowercase": bool(_cfg_get(self.config, "lowercase", True)),
            "remove_stopwords": bool(_cfg_get(self.config, "remove_stopwords", True)),
            "min_token_length": int(_cfg_get(self.config, "min_token_length", 1)),
            "token_pattern": str(_cfg_get(self.config, "token_pattern", r"[a-z0-9]+")),
        }
        self.pre_processor = PreProcessor(preprocessor_cfg)
        self.evaluator = Evaluator()

        # Runtime state populated across stages
        self.splits: dict[str, tuple[dict[str, Any], dict[str, str], dict[str, dict[str, int]]]] | None = None
        self.corpus: dict[str, Any] = {}
        self.preprocessed: dict[str, Any] | None = None
        self.retriever: Retriever | None = None
        self.spell_corrector: SpellCorrector | None = None

        self.dev_tuning_results: dict[str, Any] = {}
        self.test_rankings: dict[str, dict[str, list[tuple[str, float]]]] = {}
        self.test_evaluations: dict[str, dict[str, Any]] = {}
        self.summary: dict[str, Any] = {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "environment": _collect_environment_info(),
        }

    def _save_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def _filter_queries_by_qrels(
        queries: Mapping[str, str],
        qrels: Mapping[str, Any],
    ) -> dict[str, str]:
        """Return only queries that have relevance judgments in ``qrels``."""
        return {
            str(qid): str(queries[qid])
            for qid in sorted(qrels.keys())
            if qid in queries
        }

    def setup(self) -> dict[str, Any]:
        """Load dataset splits, build/load lexical & semantic indexes, init Retriever."""
        if self.splits is None:
            self.splits = self.data_loader.load_data()

        # Merge corpus across splits (BEIR stores the same corpus across splits)
        merged_corpus: dict[str, Any] = {}
        for split_name in ("train", "dev", "test"):
            split_tuple = self.splits.get(split_name)
            if split_tuple is not None:
                for doc_id, doc in split_tuple[0].items():
                    merged_corpus.setdefault(str(doc_id), doc)
        self.corpus = merged_corpus

        self.preprocessed = self.pre_processor.preprocess(self.splits)

        lexical_retriever = BM25Retriever(
            self.preprocessed["lexical"],
            k1=self.bm25_k1,
            b=self.bm25_b,
        )
        semantic_retriever = SemanticRetriever(
            embeddings=self.preprocessed["semantic"]["embeddings"],
            doc_ids=self.preprocessed["semantic"]["doc_ids"],
            metadata=self.preprocessed["semantic"]["metadata"],
            model_name=self.embedding_model_name,
            device=self.embedding_device,
            model=self._injected_embedding_model,
        )
        hybrid_retriever = HybridRetriever(
            k=self.rrf_k,
            fusion_depth=self.rrf_fusion_depth,
        )
        reranker = CrossEncoderReranker(
            model_name=self.reranker_model_name,
            rerank_top_k=self.rerank_top_k,
            batch_size=self.rerank_batch_size,
            device=self.reranker_device,
            model=self._injected_reranker_model,
        )
        self.retriever = Retriever(
            lexical=lexical_retriever,
            semantic=semantic_retriever,
            hybrid=hybrid_retriever,
            reranker=reranker,
            corpus=self.corpus,
        )

        self.summary["dataset_stats"] = {
            "num_documents": len(self.corpus),
            "splits": {
                split_name: {
                    "num_qrels_queries": len(self.splits[split_name][2]),
                    "num_judgments": sum(
                        len(docs) for docs in self.splits[split_name][2].values()
                    ),
                }
                for split_name in ("train", "dev", "test")
                if split_name in self.splits
            },
        }
        self.summary["indexing"] = self.preprocessed["run_metrics"]
        return self.preprocessed

    def _ensure_setup(self) -> None:
        if self.retriever is None or self.splits is None or self.preprocessed is None:
            self.setup()

    def run_dev_tuning(self) -> dict[str, Any]:
        """Tune allowed parameters strictly on the ``dev`` split (Reqs 3, 4, 8)."""
        self._ensure_setup()
        assert self.splits is not None and self.retriever is not None and self.preprocessed is not None

        _, dev_queries_raw, dev_qrels = self.splits["dev"]
        dev_queries = self._filter_queries_by_qrels(dev_queries_raw, dev_qrels)

        report: dict[str, Any] = {}

        # Optional BM25 (k1, b) tuning on dev
        if self.tune_bm25_on_dev:
            bm25_tuning = BM25Retriever.tune_on_dev(
                self.preprocessed["lexical"],
                dev_queries,
                dev_qrels,
                self.evaluator,
            )
            self.bm25_k1 = bm25_tuning["best_k1"]
            self.bm25_b = bm25_tuning["best_b"]
            self.retriever.lexical = BM25Retriever(
                self.preprocessed["lexical"],
                k1=self.bm25_k1,
                b=self.bm25_b,
            )
            report["bm25"] = bm25_tuning

        # Mandatory RRF k tuning on dev (at least k in {10, 60, 100}, Requirement 8)
        lex_dev = self.retriever.retrieve_lexical(
            dev_queries, top_k=self.rrf_fusion_depth
        )
        sem_dev = self.retriever.retrieve_semantic(
            dev_queries,
            top_k=self.rrf_fusion_depth,
            batch_size=self.semantic_batch_size,
        )
        rrf_tuning = self.retriever.hybrid.tune_k(
            lex_dev,
            sem_dev,
            dev_qrels,
            self.evaluator,
            k_candidates=self.rrf_k_candidates,
            fusion_depth=self.rrf_fusion_depth,
        )
        self.rrf_k = int(rrf_tuning["best_k"])
        report["rrf"] = rrf_tuning

        # Optional spell-corrector tuning on dev (Requirement 4)
        if self.typo_tune_on_dev:
            typo_tuning = tune_spell_corrector_on_dev(
                corpus_documents=self.corpus,
                dev_queries=dev_queries,
                dev_qrels=dev_qrels,
                lexical_retriever=self.retriever.lexical,
                evaluator=self.evaluator,
                distance_metric=self.typo_distance_metric,
                seed=self.typo_seed,
            )
            best_typo = typo_tuning["best_params"]
            self.typo_k_gram = int(best_typo["k"])
            self.typo_jaccard_threshold = float(best_typo["jaccard_threshold"])
            self.typo_max_edit_distance = int(best_typo["max_edit_distance"])
            report["typo"] = typo_tuning

        self.dev_tuning_results = report
        self.summary["dev_tuning"] = report
        self._save_json(self.dev_results_dir / "dev_tuning_results.json", report)
        return report

    def run_retrieval_pipeline(self) -> dict[str, Any]:
        """Run Lexical, Semantic, Hybrid, and Reranking on ``test`` and export CSVs."""
        self._ensure_setup()
        assert self.splits is not None and self.retriever is not None

        _, test_queries_raw, test_qrels = self.splits["test"]
        test_queries = self._filter_queries_by_qrels(test_queries_raw, test_qrels)

        all_rankings = self.retriever.retrieve_all_stages(
            test_queries,
            corpus=self.corpus,
            initial_top_k=self.rrf_fusion_depth,
            semantic_batch_size=self.semantic_batch_size,
        )
        self.test_rankings = all_rankings

        evaluations: dict[str, dict[str, Any]] = {}
        pytrec_cross_check: dict[str, dict[str, float]] = {}

        for model_name in ("lexical", "semantic", "hybrid", "reranking"):
            rankings = all_rankings[model_name]
            evaluations[model_name] = self.evaluator.evaluate(
                test_qrels,
                rankings,
                k=10,
                model_name=model_name,
            )
            self.evaluator.export_ranking_csv(
                rankings,
                test_qrels,
                self.retrieval_results_dir / f"results_{model_name}.csv",
                top_k=100,
            )

            # Verify metrics against pytrec_eval
            trec_per_q = self.evaluator.cross_check_with_pytrec_eval(
                test_qrels, rankings
            )
            n_q = len(trec_per_q)
            pytrec_cross_check[model_name] = {
                "Precision@10": sum(m["P_10"] for m in trec_per_q.values()) / n_q,
                "Recall@10": sum(m["recall_10"] for m in trec_per_q.values()) / n_q,
                "MAP@10": sum(m["map_cut_10"] for m in trec_per_q.values()) / n_q,
                "NDCG@10": sum(m["ndcg_cut_10"] for m in trec_per_q.values()) / n_q,
            }

        self.test_evaluations = evaluations
        self.evaluator.export_metrics_per_query_csv(
            evaluations,
            self.retrieval_results_dir / "metrics_per_query.csv",
        )

        # Requirement 7: Lexical vs. Semantic per-query analysis with concrete evidence
        lex_vs_sem_raw = self.evaluator.compare_lexical_vs_semantic(
            test_qrels,
            all_rankings["lexical"],
            all_rankings["semantic"],
            top_n_queries=3,
            top_docs_per_query=5,
        )
        lex_vs_sem_enriched = enrich_lexical_vs_semantic_analysis(
            lex_vs_sem_raw,
            test_queries,
            self.corpus,
            self.retriever.lexical,
            all_rankings["lexical"],
            all_rankings["semantic"],
            test_qrels,
        )
        self._save_json(
            self.retrieval_results_dir / "analysis_lexical_vs_semantic.json",
            lex_vs_sem_enriched,
        )

        # Requirement 10: Reranking movement analysis with concrete evidence
        rerank_raw = self.evaluator.analyze_reranking(
            test_qrels,
            all_rankings["hybrid"],
            all_rankings["reranking"],
            top_gains=2,
        )
        rerank_enriched = enrich_reranking_analysis(
            rerank_raw,
            test_queries,
            self.corpus,
            all_rankings["hybrid"],
            all_rankings["reranking"],
            test_qrels,
        )
        self._save_json(
            self.retrieval_results_dir / "analysis_reranking.json",
            rerank_enriched,
        )

        retrieval_summary = {
            "global_metrics": {
                name: ev["global"] for name, ev in evaluations.items()
            },
            "pytrec_eval_cross_check": pytrec_cross_check,
            "models_and_timings": self.retriever.get_metadata(),
        }
        self.summary["retrieval"] = retrieval_summary
        return retrieval_summary

    def _get_or_load_rankings(
        self,
        model_name: str,
    ) -> dict[str, list[tuple[str, float]]]:
        if model_name in self.test_rankings:
            return self.test_rankings[model_name]
        csv_path = self.retrieval_results_dir / f"results_{model_name}.csv"
        legacy_path = self.results_dir / csv_path.name
        # Fall back only for flat runs, so an incomplete new run cannot mix rankings.
        if not self.retrieval_results_dir.exists() and legacy_path.is_file():
            csv_path = legacy_path
        if csv_path.is_file():
            loaded = _load_ranking_csv(csv_path)
            self.test_rankings[model_name] = loaded
            return loaded
        raise RuntimeError(
            f"Rankings for '{model_name}' are not available in memory or at {csv_path}. "
            "Run the 'retrieval' stage first."
        )

    def _build_spell_corrector(self) -> SpellCorrector:
        corrector = SpellCorrector(
            k=self.typo_k_gram,
            jaccard_threshold=self.typo_jaccard_threshold,
            max_edit_distance=self.typo_max_edit_distance,
            distance_metric=self.typo_distance_metric,
        )
        corrector.fit_from_corpus(self.corpus)
        self.spell_corrector = corrector
        return corrector

    def run_typo_experiment(
        self,
        custom_cases: Sequence[Mapping[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run Requirement 4 spelling-correction experiment on 10 test queries."""
        self._ensure_setup()
        assert self.splits is not None and self.retriever is not None

        _, test_queries_raw, test_qrels = self.splits["test"]
        test_queries = self._filter_queries_by_qrels(test_queries_raw, test_qrels)

        corrector = self._build_spell_corrector()

        # Load custom typo definitions if supplied via argument or config path
        typo_cases: list[dict[str, str]]
        if custom_cases is not None:
            typo_cases = [dict(c) for c in custom_cases]
        elif self.typo_queries_path is not None:
            with open(self.typo_queries_path, "r", encoding="utf-8") as f:
                typo_cases = json.load(f)
        else:
            # Ensure original lexical NDCG@10 > 0 for every selected query (Req 4-(iii))
            try:
                lexical_rankings = self._get_or_load_rankings("lexical")
            except RuntimeError:
                lexical_rankings = self.retriever.retrieve_lexical(
                    test_queries, top_k=100
                )
            lex_eval = self.evaluator.evaluate(
                test_qrels, lexical_rankings, k=10, model_name="lexical"
            )
            eligible_qids = [
                qid
                for qid, m in lex_eval["per_query"].items()
                if m["NDCG10"] > 0.0
            ]
            typo_cases = build_typo_benchmark(
                test_queries,
                eligible_qids,
                corrector.vocabulary,
                num_queries=self.typo_num_queries,
                seed=self.typo_seed,
            )

        lexical_records: list[dict[str, Any]] = []
        semantic_records: list[dict[str, Any]] = []
        detailed_logs: list[dict[str, Any]] = []

        for case in typo_cases:
            qid = str(case["query_id"])
            q_orig = str(case.get("query_original") or test_queries[qid])
            q_err = str(case["query_erro"])
            tipo_erro = str(case["tipo_erro"])

            validation = corrector.validate_typo_pair(q_orig, q_err)
            if not validation["is_valid"]:
                raise ValueError(
                    f"Requirement 4-(ii) violated for {qid}: typo query '{q_err}' "
                    f"does not introduce an unknown OOV term (validation={validation})."
                )

            corr_details = corrector.correct_query_with_details(q_err)
            q_corr = corr_details["corrected_query"]

            lex_orig = self.retriever.lexical.search(q_orig, top_k=100)
            lex_err = self.retriever.lexical.search(q_err, top_k=100)
            lex_corr = self.retriever.lexical.search(q_corr, top_k=100)

            lexical_records.append(
                {
                    "query_id": qid,
                    "query_original": q_orig,
                    "query_erro": q_err,
                    "query_corrigida": q_corr,
                    "tipo_erro": tipo_erro,
                    "ranking_original": lex_orig,
                    "ranking_erro": lex_err,
                    "ranking_corrigida": lex_corr,
                }
            )

            sem_entry: dict[str, Any] | None = None
            if self.typo_eval_semantic:
                sem_orig = self.retriever.semantic.search(q_orig, top_k=100)
                sem_err = self.retriever.semantic.search(q_err, top_k=100)
                sem_corr = self.retriever.semantic.search(q_corr, top_k=100)
                q_qrels = test_qrels[qid]
                ndcg_s_orig = self.evaluator.evaluate_query(q_qrels, sem_orig, k=10)["NDCG10"]
                ndcg_s_err = self.evaluator.evaluate_query(q_qrels, sem_err, k=10)["NDCG10"]
                ndcg_s_corr = self.evaluator.evaluate_query(q_qrels, sem_corr, k=10)["NDCG10"]
                sem_rec = self.evaluator.compute_typo_recovery(
                    ndcg_s_orig, ndcg_s_err, ndcg_s_corr
                )
                sem_entry = {
                    "query_id": qid,
                    "query_original": q_orig,
                    "query_erro": q_err,
                    "query_corrigida": q_corr,
                    "tipo_erro": tipo_erro,
                    "NDCG10_original": ndcg_s_orig,
                    "NDCG10_erro": ndcg_s_err,
                    "NDCG10_corrigida": ndcg_s_corr,
                    **sem_rec,
                }
                semantic_records.append(sem_entry)

            detailed_logs.append(
                {
                    "query_id": qid,
                    "tipo_erro": tipo_erro,
                    "validation": validation,
                    "correction_details": corr_details,
                    "semantic_robustness": sem_entry,
                }
            )

        evaluated_lexical = self.evaluator.evaluate_typo_experiment(
            test_qrels, lexical_records
        )
        self.evaluator.export_typo_experiment_csv(
            evaluated_lexical,
            self.typo_results_dir / "typo_experiment.csv",
        )
        if semantic_records:
            self.evaluator.export_typo_experiment_csv(
                semantic_records,
                self.typo_results_dir / "typo_experiment_semantic.csv",
            )

        valid_recoveries = [
            r["recovery_rate"]
            for r in evaluated_lexical
            if r["recovery_rate"] is not None
        ]
        typo_summary = {
            "corrector_metadata": corrector.get_metadata(),
            "num_queries": len(evaluated_lexical),
            "mean_ndcg10_original": sum(
                r["NDCG10_original"] for r in evaluated_lexical
            )
            / len(evaluated_lexical),
            "mean_ndcg10_erro": sum(r["NDCG10_erro"] for r in evaluated_lexical)
            / len(evaluated_lexical),
            "mean_ndcg10_corrigida": sum(
                r["NDCG10_corrigida"] for r in evaluated_lexical
            )
            / len(evaluated_lexical),
            "queries_with_degradation": len(valid_recoveries),
            "mean_recovery_rate": (
                sum(valid_recoveries) / len(valid_recoveries)
                if valid_recoveries
                else None
            ),
            "records": evaluated_lexical,
            "semantic_optional": semantic_records if semantic_records else None,
            "details": detailed_logs,
        }
        self._save_json(
            self.typo_results_dir / "typo_experiment_details.json", typo_summary
        )
        self.summary["typo_experiment"] = {
            k: v
            for k, v in typo_summary.items()
            if k not in ("details",)
        }
        return typo_summary

    def run_rag_experiment(self) -> dict[str, Any]:
        """Run Requirements 11-14 RAG A/B experiment on the 10 fixed test queries."""
        self._ensure_setup()
        assert self.splits is not None

        _, test_queries_raw, test_qrels = self.splits["test"]
        test_queries = self._filter_queries_by_qrels(test_queries_raw, test_qrels)

        # Select the fixed 10 test question queries with seed 42 (Requirement 12)
        rag_query_ids = self.evaluator.select_rag_queries(
            test_qrels,
            test_queries,
            seed=self.rag_seed,
            sample_size=self.rag_sample_size,
        )
        if (
            self.dataset_name.lower() == "nfcorpus"
            and self.rag_seed == 42
            and self.rag_sample_size == 10
            and len(test_qrels) == 323
            and tuple(rag_query_ids) != EXPECTED_NFCORPUS_RAG_QUERIES
        ):
            raise RuntimeError(
                f"Selected RAG query IDs {rag_query_ids} do not match the "
                f"specification list {list(EXPECTED_NFCORPUS_RAG_QUERIES)}."
            )

        lexical_rankings = self._get_or_load_rankings("lexical")
        semantic_rankings = self._get_or_load_rankings("semantic")
        hybrid_rankings = self._get_or_load_rankings("hybrid")
        reranked_rankings = self._get_or_load_rankings("reranking")

        # Evaluate retrieval at @5 for the 10 RAG queries (Requirement 13)
        top5_rankings = {
            qid: reranked_rankings.get(qid, [])[: self.rag_top_k]
            for qid in rag_query_ids
        }
        rag_retrieval_eval = self.evaluator.evaluate_rag_retrieval(
            test_qrels,
            top5_rankings,
            query_ids=rag_query_ids,
        )

        rag_llm = RAGLLM(
            model=self.llm_model,
            temperature=self.llm_temperature,
            max_output_tokens=self.llm_max_output_tokens,
            seed=self.llm_seed,
            language=self.llm_language,
            api_key=self.google_api_key,
            top_k_context=self.rag_top_k,
            provider=self.llm_provider,
            load_in_4bit=self.llm_load_in_4bit,
            enable_thinking=self.llm_enable_thinking,
            device_map=self.llm_device_map,
            llm=self._injected_llm_client,
        )

        rag_records: list[dict[str, Any]] = []
        latencies_a: list[float] = []
        latencies_b: list[float] = []

        for qid in rag_query_ids:
            qtext = test_queries[qid]
            top5_docs = []
            for doc_id, score in top5_rankings[qid]:
                raw_doc = self.corpus.get(doc_id, {})
                if isinstance(raw_doc, Mapping):
                    top5_docs.append(
                        {
                            "doc_id": doc_id,
                            "title": raw_doc.get("title", ""),
                            "text": raw_doc.get("text", ""),
                            "score": score,
                        }
                    )
                else:
                    top5_docs.append(
                        {"doc_id": doc_id, "text": str(raw_doc), "score": score}
                    )

            # Condition A: without RAG
            resp_a = rag_llm.generate_with_metadata(query=qtext, documents=None)
            latencies_a.append(resp_a.latency_seconds)
            rag_records.append(
                {
                    "query_id": qid,
                    "query": qtext,
                    "condition": "A",
                    "context_doc_ids": [],
                    "prompt": resp_a.prompt,
                    "answer": resp_a.answer,
                    "fidelidade": "",
                    "justificativa_fidelidade": "",
                    "completude": "",
                    "justificativa_completude": "",
                    "utilidade": "",
                    "justificativa_utilidade": "",
                    "rastreabilidade": "N/A",
                    "justificativa_rastreabilidade": (
                        "Condicao sem RAG: nenhum contexto externo fornecido."
                    ),
                }
            )

            # Condition B: with RAG (Top-5 reranked documents)
            resp_b = rag_llm.generate_with_metadata(
                query=qtext,
                documents=top5_docs,
                top_k=self.rag_top_k,
            )
            latencies_b.append(resp_b.latency_seconds)
            rag_records.append(
                {
                    "query_id": qid,
                    "query": qtext,
                    "condition": "B",
                    "context_doc_ids": resp_b.context_doc_ids,
                    "prompt": resp_b.prompt,
                    "answer": resp_b.answer,
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

        # Export standardized rag_experiment.csv (Requirement 15)
        rag_csv_path = self.evaluator.export_rag_experiment_csv(
            rag_records,
            self.rag_results_dir / "rag_experiment.csv",
        )

        # Export blinded evaluation template + key + context reference (Requirement 13)
        blind_info = build_rag_blind_evaluation(
            rag_records,
            blind_csv_path=self.human_evaluation_dir / "rag_blind_evaluation.csv",
            key_json_path=self.human_evaluation_dir / "rag_blind_key.json",
            corpus=self.corpus,
            context_reference_path=self.human_evaluation_dir / "rag_context_reference.json",
            seed=self.rag_seed,
        )

        # Export end-to-end 6-stage pipeline trace for failure localization (Requirement 14)
        pipeline_trace = build_end_to_end_rag_trace(
            rag_query_ids=rag_query_ids,
            queries=test_queries,
            corpus=self.corpus,
            qrels=test_qrels,
            lexical_rankings=lexical_rankings,
            semantic_rankings=semantic_rankings,
            hybrid_rankings=hybrid_rankings,
            reranked_rankings=reranked_rankings,
            rag_records=rag_records,
            top_k_context=self.rag_top_k,
        )
        trace_path = self._save_json(
            self.rag_results_dir / "rag_pipeline_trace.json",
            pipeline_trace,
        )

        rag_summary = {
            "query_ids": rag_query_ids,
            "llm_config": {
                "model": rag_llm.model,
                "provider": rag_llm._resolved_provider(),
                "load_in_4bit": rag_llm.load_in_4bit,
                "enable_thinking": rag_llm.enable_thinking,
                "temperature": rag_llm.temperature,
                "max_output_tokens": rag_llm.max_output_tokens,
                "seed": rag_llm.seed,
                "language": rag_llm.language,
                "top_k_context": rag_llm.top_k_context,
            },
            "retrieval_metrics_at_5": rag_retrieval_eval,
            "generation_latency": {
                "condition_A_mean_seconds": sum(latencies_a) / len(latencies_a),
                "condition_B_mean_seconds": sum(latencies_b) / len(latencies_b),
                "condition_A_total_seconds": sum(latencies_a),
                "condition_B_total_seconds": sum(latencies_b),
            },
            "files": {
                "rag_experiment_csv": str(rag_csv_path),
                "pipeline_trace_json": str(trace_path),
                **blind_info,
            },
        }
        self._save_json(self.rag_results_dir / "rag_experiment_summary.json", rag_summary)
        self.summary["rag_experiment"] = rag_summary
        return rag_summary

    def aggregate_rag_scores(
        self,
        blind_csv_paths: Sequence[str | Path],
        key_json_path: str | Path | None = None,
        rag_csv_path: str | Path | None = None,
        output_csv_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Unblind human RAG scores, update ``rag_experiment.csv``, and compute averages/Kappa."""
        rag_dir = self.rag_results_dir
        evaluation_dir = self.human_evaluation_dir
        if not rag_dir.exists() and (self.results_dir / "rag_experiment.csv").is_file():
            rag_dir = evaluation_dir = self.results_dir
        rag_csv = Path(rag_csv_path) if rag_csv_path else rag_dir / "rag_experiment.csv"
        key_json = Path(key_json_path) if key_json_path else evaluation_dir / "rag_blind_key.json"
        result = merge_blind_evaluations(
            rag_csv_path=rag_csv,
            blind_csv_paths=blind_csv_paths,
            key_json_path=key_json,
            output_csv_path=output_csv_path or rag_csv,
        )
        self._save_json(
            evaluation_dir / "rag_human_evaluation_summary.json",
            result,
        )
        self.summary["rag_human_evaluation"] = result
        return result

    def _persist_summary(self) -> Path:
        """Merge current summary into ``experiment_summary.json`` and save."""
        summary_path = self.results_dir / "experiment_summary.json"
        merged: dict[str, Any] = {}
        if summary_path.is_file():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if isinstance(existing, dict):
                    merged.update(existing)
            except (OSError, json.JSONDecodeError):
                pass
        merged.update(self.summary)
        self.summary = merged
        return self._save_json(summary_path, merged)

    def run(
        self,
        stages: Sequence[str] = ("setup", "dev-tuning", "retrieval", "typo", "rag"),
    ) -> dict[str, Any]:
        """Execute one or more pipeline stages in order and save consolidated summary."""
        normalized_stages = [s.strip().lower() for s in stages]
        if "all" in normalized_stages:
            normalized_stages = ["setup", "dev-tuning", "retrieval", "typo", "rag"]

        for stage in normalized_stages:
            if stage == "setup":
                self.setup()
            elif stage in ("dev-tuning", "dev_tuning", "tune"):
                self.run_dev_tuning()
            elif stage == "retrieval":
                self.run_retrieval_pipeline()
            elif stage == "typo":
                self.run_typo_experiment()
            elif stage == "rag":
                self.run_rag_experiment()
            else:
                raise ValueError(
                    f"Unknown stage '{stage}'. Valid stages: "
                    "'all', 'setup', 'dev-tuning', 'retrieval', 'typo', 'rag'."
                )
            self._persist_summary()

        return self.summary
