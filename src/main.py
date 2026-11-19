"""Command-line entry point for running Trabalho Pratico 2 experiments.

Examples
--------
Run the full pipeline (dev tuning, test retrieval + reranking, typo experiment,
and RAG A/B experiment):
    python -m src.main --stages all

Run only indexing, dev tuning, test retrieval/reranking, and typo experiment
(no LLM API key required):
    python -m src.main --stages setup dev-tuning retrieval typo

Run only the RAG experiment (reusing saved ``results_*.csv`` rankings):
    python -m src.main --stages rag --llm-language Portuguese

Aggregate completed blinded human evaluations back into ``rag_experiment.csv``:
    python -m src.main --stages rag-aggregate \\
        --blind-csv results/rag_blind_eval_annotator1.csv results/rag_blind_eval_annotator2.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Support both `python -m src.main` and `python src/main.py`
_ROOT_DIR = Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from src.experiment import Experiment
from src.pre_processing import DEFAULT_EMBEDDING_MODEL
from src.retrieval import DEFAULT_QWEN_MODEL, DEFAULT_RERANKER_MODEL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trabalho Pratico 2 (SCC0282 - IR): Lexical, Semantic, Hybrid (RRF), "
            "Cross-Encoder Reranking, Typo Tolerance, and RAG pipeline."
        )
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_name",
        default="nfcorpus",
        help="BEIR dataset name (default: nfcorpus).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=[
            "all",
            "setup",
            "dev-tuning",
            "retrieval",
            "typo",
            "rag",
            "rag-aggregate",
        ],
        help="Pipeline stages to execute in order (default: all).",
    )
    parser.add_argument("--data-dir", default="datasets", help="BEIR datasets directory.")
    parser.add_argument("--cache-dir", default="cache", help="Index cache directory.")
    parser.add_argument("--runs-dir", default="runs", help="Run metrics directory.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for standardized CSV and analysis outputs.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore existing lexical/semantic cache files and rebuild from scratch.",
    )

    # Lexical (BM25)
    parser.add_argument("--bm25-k1", type=float, default=1.5, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=0.75, help="BM25 b parameter.")
    parser.add_argument(
        "--tune-bm25",
        dest="tune_bm25_on_dev",
        action="store_true",
        help="Tune BM25 (k1, b) on the dev split.",
    )

    # Semantic (Bi-Encoder)
    parser.add_argument(
        "--embedding-model",
        dest="embedding_model_name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence Transformers bi-encoder model (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--embedding-device",
        default=None,
        help="Optional torch device for embeddings (e.g., cpu, cuda).",
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=None,
        help="Optional batch size for query encoding (default: None = per-query latency).",
    )

    # Hybrid (RRF)
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Default RRF k when dev-tuning is skipped (default: 60).",
    )
    parser.add_argument(
        "--rrf-k-candidates",
        nargs="+",
        type=int,
        default=[10, 60, 100],
        help="RRF k candidates evaluated on dev (default: 10 60 100).",
    )
    parser.add_argument(
        "--rrf-fusion-depth",
        type=int,
        default=100,
        help="Top-N docs per system fused by RRF (default: 100).",
    )

    # Reranking (Cross-Encoder)
    parser.add_argument(
        "--reranker-model",
        dest="reranker_model_name",
        default=DEFAULT_RERANKER_MODEL,
        help=f"Cross-Encoder reranker model (default: {DEFAULT_RERANKER_MODEL}).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=50,
        help="Number of top hybrid docs reordered by Cross-Encoder (default: 50).",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=32,
        help="Batch size for Cross-Encoder inference (default: 32).",
    )
    parser.add_argument(
        "--reranker-device",
        default=None,
        help="Optional torch device for cross-encoder (e.g., cpu, cuda).",
    )

    # Typo experiment
    parser.add_argument(
        "--typo-k-gram",
        type=int,
        default=2,
        help="Character k-gram size for spelling correction (default: 2).",
    )
    parser.add_argument(
        "--typo-jaccard-threshold",
        type=float,
        default=0.25,
        help="Minimum Jaccard similarity for typo candidates (default: 0.25).",
    )
    parser.add_argument(
        "--typo-max-edit-distance",
        type=int,
        default=2,
        help="Maximum edit distance for typo selection (default: 2).",
    )
    parser.add_argument(
        "--no-tune-typo",
        dest="typo_tune_on_dev",
        action="store_false",
        help="Skip dev-set tuning of typo corrector hyperparameters.",
    )
    parser.add_argument(
        "--typo-queries-json",
        dest="typo_queries_path",
        default=None,
        help="Optional JSON file with 10 custom typo queries.",
    )
    parser.add_argument(
        "--no-typo-semantic",
        dest="typo_eval_semantic",
        action="store_false",
        help="Skip optional evaluation of typo queries on the semantic retriever.",
    )

    # RAG experiment
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_QWEN_MODEL,
        help=f"LLM model identifier for RAG (default: {DEFAULT_QWEN_MODEL}).",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "local", "gemini"],
        default="auto",
        help="LLM backend provider (default: auto; uses local 4-bit model for Qwen).",
    )
    parser.add_argument(
        "--no-llm-4bit",
        dest="llm_load_in_4bit",
        action="store_false",
        help="Disable 4-bit NF4 quantization when running a local Transformers model.",
    )
    parser.add_argument(
        "--llm-enable-thinking",
        action="store_true",
        help="Enable Qwen3 <think> reasoning mode during generation (default: disabled).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--llm-max-output-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens per LLM answer (default: 1024).",
    )
    parser.add_argument(
        "--llm-seed",
        type=int,
        default=42,
        help="Random seed passed to the LLM (default: 42).",
    )
    parser.add_argument(
        "--llm-language",
        default="Portuguese",
        help="Answer language enforced for Conditions A and B (default: Portuguese).",
    )

    # Blind RAG human-evaluation aggregation
    parser.add_argument(
        "--blind-csv",
        nargs="+",
        default=None,
        help="One or two completed blind evaluation CSV files for rag-aggregate.",
    )
    parser.add_argument(
        "--blind-key",
        default=None,
        help="Path to rag_blind_key.json (default: <results-dir>/rag_blind_key.json).",
    )

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = vars(args)
    experiment = Experiment(config)

    stages = list(args.stages)
    if "rag-aggregate" in stages:
        other_stages = [s for s in stages if s != "rag-aggregate"]
        if other_stages:
            experiment.run(other_stages)
        if not args.blind_csv:
            raise SystemExit(
                "Error: --blind-csv is required when running the 'rag-aggregate' stage."
            )
        experiment.aggregate_rag_scores(
            blind_csv_paths=args.blind_csv,
            key_json_path=args.blind_key,
        )
        return

    experiment.run(stages)


if __name__ == "__main__":
    main()
