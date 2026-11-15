"""Unified preprocessing, indexing, and caching orchestrator.

Coordinates single-use lexical (:class:`InvertedIndexer`) and semantic
(:class:`EmbeddingIndexer`) indexers, persisting caches as ``.json`` and
``.npy`` files under ``cache/`` and recording cache build/load timing metrics
in the active run directory under ``runs/``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .lexical_index import InvertedIndexer
from .semantic_index import DEFAULT_EMBEDDING_MODEL, EmbeddingIndexer


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    """Read a key from Config, dict/Mapping, argparse.Namespace, or None."""
    if config is None:
        return default
    if hasattr(config, "get"):
        value = config.get(key, default)
        return default if value is None else value
    value = getattr(config, key, default)
    return default if value is None else value


def _file_is_non_empty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _merge_corpus_entry(
    target: dict[str, str],
    doc_id: str,
    raw_doc: Any,
) -> None:
    if isinstance(raw_doc, Mapping):
        title = str(raw_doc.get("title") or "").strip()
        text = str(raw_doc.get("text") or "").strip()
        full_text = f"{title} {text}".strip() if (title and text) else (title or text)
    else:
        full_text = str(raw_doc).strip()
    target[str(doc_id)] = full_text


def _extract_documents(data: Any) -> dict[str, str]:
    """Extract a ``{doc_id: "title text"}`` mapping from BEIR/DataLoader inputs."""
    if data is None:
        raise ValueError("data cannot be None when building indexes from scratch.")

    documents: dict[str, str] = {}

    # Case 1: Tuple (corpus, queries, qrels)
    if isinstance(data, tuple) and len(data) >= 1 and isinstance(data[0], Mapping):
        for doc_id, doc in data[0].items():
            _merge_corpus_entry(documents, str(doc_id), doc)
        return documents

    # Case 2: Dict returned by DataLoader.load_data() -> {"train": (corpus, ...), ...}
    if isinstance(data, Mapping):
        split_keys = ("train", "val", "dev", "test")
        if any(
            k in data and isinstance(data[k], tuple) and len(data[k]) >= 1
            for k in split_keys
        ):
            for split in split_keys:
                split_val = data.get(split)
                if isinstance(split_val, tuple) and isinstance(split_val[0], Mapping):
                    for doc_id, doc in split_val[0].items():
                        if str(doc_id) not in documents:
                            _merge_corpus_entry(documents, str(doc_id), doc)
            return documents

        # Case 3: Direct BEIR corpus mapping {doc_id: {"title": ..., "text": ...}}
        for doc_id, doc in data.items():
            _merge_corpus_entry(documents, str(doc_id), doc)
        return documents

    # Case 4: Sequence of document dicts [{"doc_id"|"_id"|"id": ..., "title": ..., "text": ...}]
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        for idx, item in enumerate(data):
            if not isinstance(item, Mapping):
                raise TypeError(
                    f"Unsupported document item at index {idx}: {type(item).__name__}"
                )
            doc_id = item.get("doc_id") or item.get("_id") or item.get("id")
            if doc_id is None:
                raise ValueError(f"Document at index {idx} is missing a doc_id/_id/id.")
            _merge_corpus_entry(documents, str(doc_id), item)
        return documents

    raise TypeError(f"Unsupported data format for PreProcessor: {type(data).__name__}")


class PreProcessor:
    """Builds or loads cached lexical and semantic indexes and logs run costs."""

    def __init__(self, config: Any = None) -> None:
        self.config = config
        self.cache_dir = Path(_cfg_get(config, "cache_dir", "cache"))
        self.lexical_cache_path = self.cache_dir / _cfg_get(
            config, "lexical_cache_file", "lexical_index.json"
        )
        self.semantic_embeddings_path = self.cache_dir / _cfg_get(
            config, "semantic_embeddings_file", "semantic_embeddings.npy"
        )
        self.semantic_metadata_path = self.cache_dir / _cfg_get(
            config, "semantic_metadata_file", "semantic_metadata.json"
        )
        self.force_rebuild = bool(_cfg_get(config, "force_rebuild", False))

        # Run directory setup for logging cache creation and retrieval costs
        explicit_run_dir = _cfg_get(config, "run_dir", None)
        if explicit_run_dir is not None:
            self.run_dir = Path(explicit_run_dir)
            self.run_id = self.run_dir.name
        else:
            runs_dir = Path(_cfg_get(config, "runs_dir", "runs"))
            self.run_id = str(
                _cfg_get(
                    config,
                    "run_id",
                    datetime.now().strftime("run_%Y%m%d_%H%M%S"),
                )
            )
            self.run_dir = runs_dir / self.run_id
        self.run_metrics_path = self.run_dir / "index_cache_metrics.json"

    def _try_load_lexical_cache(self) -> tuple[dict[str, Any] | None, float | None]:
        """Load lexical JSON cache if it exists and is non-empty."""
        if self.force_rebuild or not _file_is_non_empty(self.lexical_cache_path):
            return None, None

        start = time.perf_counter()
        try:
            with open(self.lexical_cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None, None
        load_time = time.perf_counter() - start

        if (
            not isinstance(payload, dict)
            or not payload.get("index")
            or not payload.get("doc_lengths")
            or int(payload.get("N", 0)) <= 0
        ):
            return None, None

        return payload, load_time

    def _try_load_semantic_cache(
        self,
    ) -> tuple[np.ndarray | None, dict[str, Any] | None, float | None]:
        """Load semantic ``.npy`` + ``.json`` cache if both exist and are non-empty."""
        if self.force_rebuild or not (
            _file_is_non_empty(self.semantic_embeddings_path)
            and _file_is_non_empty(self.semantic_metadata_path)
        ):
            return None, None, None

        start = time.perf_counter()
        try:
            with open(self.semantic_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            with open(self.semantic_embeddings_path, "rb") as f:
                embeddings = np.load(f, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError):
            return None, None, None
        load_time = time.perf_counter() - start

        doc_ids = metadata.get("doc_ids") if isinstance(metadata, dict) else None
        if (
            not isinstance(doc_ids, list)
            or len(doc_ids) == 0
            or not isinstance(embeddings, np.ndarray)
            or embeddings.ndim != 2
            or embeddings.size == 0
            or embeddings.shape[0] != len(doc_ids)
        ):
            return None, None, None

        return embeddings, metadata, load_time

    def _build_and_cache_lexical(
        self, documents: Mapping[str, str]
    ) -> tuple[dict[str, Any], float]:
        indexer = InvertedIndexer(
            lowercase=bool(_cfg_get(self.config, "lowercase", True)),
            remove_stopwords=bool(_cfg_get(self.config, "remove_stopwords", True)),
            min_token_length=int(_cfg_get(self.config, "min_token_length", 1)),
            token_pattern=str(_cfg_get(self.config, "token_pattern", r"[a-z0-9]+")),
        )
        payload = indexer.build(documents)

        self.lexical_cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_start = time.perf_counter()
        with open(self.lexical_cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        save_time = time.perf_counter() - save_start
        return payload, save_time

    def _build_and_cache_semantic(
        self, documents: Mapping[str, str]
    ) -> tuple[np.ndarray, dict[str, Any], float]:
        indexer = EmbeddingIndexer(
            embedding_model_name=str(
                _cfg_get(self.config, "embedding_model_name", DEFAULT_EMBEDDING_MODEL)
            ),
            batch_size=int(_cfg_get(self.config, "embedding_batch_size", 64)),
            normalize_embeddings=bool(
                _cfg_get(self.config, "normalize_embeddings", True)
            ),
            device=_cfg_get(self.config, "embedding_device", None),
            show_progress_bar=bool(
                _cfg_get(self.config, "embedding_show_progress_bar", False)
            ),
            model=_cfg_get(self.config, "embedding_model", None),
        )
        embeddings, metadata = indexer.index(documents)

        self.semantic_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self.semantic_metadata_path.parent.mkdir(parents=True, exist_ok=True)

        save_start = time.perf_counter()
        with open(self.semantic_embeddings_path, "wb") as f:
            np.save(f, embeddings)
        with open(self.semantic_metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        save_time = time.perf_counter() - save_start
        return embeddings, metadata, save_time

    @staticmethod
    def _file_sizes(paths: list[Path]) -> dict[str, int]:
        return {
            str(p): (p.stat().st_size if p.exists() else 0)
            for p in paths
        }

    def _record_run_metrics(
        self,
        lexical_metrics: dict[str, Any],
        semantic_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        existing_history: list[dict[str, Any]] = []
        if _file_is_non_empty(self.run_metrics_path):
            try:
                with open(self.run_metrics_path, "r", encoding="utf-8") as f:
                    prev = json.load(f)
                if isinstance(prev, dict) and isinstance(prev.get("history"), list):
                    existing_history = prev["history"]
            except (OSError, json.JSONDecodeError):
                existing_history = []

        event = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "lexical": lexical_metrics,
            "semantic": semantic_metrics,
        }
        existing_history.append(event)

        report = {
            "run_id": self.run_id,
            "updated_at": event["timestamp"],
            "cache_dir": str(self.cache_dir),
            "lexical": lexical_metrics,
            "semantic": semantic_metrics,
            "history": existing_history,
        }
        with open(self.run_metrics_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    def _format_index_metrics(
        self,
        *,
        cache_hit: bool,
        build_time: float | None,
        save_time: float | None,
        load_time: float | None,
        cache_files: list[Path],
        stats: dict[str, Any],
    ) -> dict[str, Any]:
        total_build_and_cache = (
            (float(build_time) + float(save_time))
            if (not cache_hit and build_time is not None and save_time is not None)
            else None
        )
        return {
            "status": "loaded_from_cache" if cache_hit else "built_and_cached",
            "cache_hit": cache_hit,
            "cache_load_time_seconds": load_time if cache_hit else None,
            "index_build_time_seconds": build_time,
            "cache_save_time_seconds": save_time if not cache_hit else None,
            "total_build_and_cache_time_seconds": total_build_and_cache,
            "cache_files_bytes": self._file_sizes(cache_files),
            "stats": stats,
        }

    def preprocess(self, data: Any = None) -> dict[str, Any]:
        """Load lexical & semantic indexes from cache, or build and cache them.

        If non-empty ``.json`` and ``.npy`` cache files exist (and
        ``force_rebuild`` is False), the single-use indexers are not executed.
        Costs for caching (build + save) and retrieving from cache (load) are
        recorded in ``runs/<run_id>/index_cache_metrics.json``.
        """
        lexical_data, lexical_load_time = self._try_load_lexical_cache()
        embeddings, semantic_metadata, semantic_load_time = (
            self._try_load_semantic_cache()
        )

        documents: dict[str, str] | None = None
        if lexical_data is None or embeddings is None or semantic_metadata is None:
            documents = _extract_documents(data)
            if not documents:
                raise ValueError("Extracted corpus is empty; cannot build indexes.")

        lexical_cache_hit = lexical_data is not None
        lexical_save_time: float | None = None
        if not lexical_cache_hit:
            assert documents is not None
            lexical_data, lexical_save_time = self._build_and_cache_lexical(documents)

        lexical_metrics = self._format_index_metrics(
            cache_hit=lexical_cache_hit,
            build_time=lexical_data.get("build_time_seconds"),
            save_time=lexical_save_time,
            load_time=lexical_load_time,
            cache_files=[self.lexical_cache_path],
            stats={
                "N": lexical_data["N"],
                "vocab_size": lexical_data["vocab_size"],
                "total_postings": lexical_data["total_postings"],
                "avg_doc_length": lexical_data["avg_doc_length"],
                "params": lexical_data.get("params"),
            },
        )

        semantic_cache_hit = embeddings is not None and semantic_metadata is not None
        semantic_save_time: float | None = None
        if not semantic_cache_hit:
            assert documents is not None
            embeddings, semantic_metadata, semantic_save_time = (
                self._build_and_cache_semantic(documents)
            )

        assert semantic_metadata is not None
        semantic_metrics = self._format_index_metrics(
            cache_hit=semantic_cache_hit,
            build_time=semantic_metadata.get("build_time_seconds"),
            save_time=semantic_save_time,
            load_time=semantic_load_time,
            cache_files=[self.semantic_embeddings_path, self.semantic_metadata_path],
            stats={
                "num_documents": semantic_metadata["num_documents"],
                "model_name": semantic_metadata["model_name"],
                "library": semantic_metadata.get("library"),
                "embedding_dim": semantic_metadata["embedding_dim"],
                "similarity_function": semantic_metadata.get("similarity_function"),
                "normalized": semantic_metadata.get("normalized"),
                "max_seq_length": semantic_metadata.get("max_seq_length"),
                "truncated_docs_count": semantic_metadata.get("truncated_docs_count"),
                "had_truncation": semantic_metadata.get("had_truncation"),
            },
        )

        run_metrics = self._record_run_metrics(lexical_metrics, semantic_metrics)

        return {
            "lexical": lexical_data,
            "semantic": {
                "embeddings": embeddings,
                "doc_ids": semantic_metadata["doc_ids"],
                "metadata": semantic_metadata,
            },
            "run_metrics": run_metrics,
            "raw_data": data,
        }
