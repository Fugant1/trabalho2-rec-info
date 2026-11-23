import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pre_processing import PreProcessor
from src.pre_processing.lexical_index import InvertedIndexer
from src.pre_processing.semantic_index import DEFAULT_EMBEDDING_MODEL, EmbeddingIndexer


class _DummyTokenizer:
    def __call__(self, texts, truncation=False, add_special_tokens=True, verbose=False):
        return {"input_ids": [[1] * (len(t.split()) + 2) for t in texts]}


class _DummySentenceTransformer:
    def __init__(self, dim: int = 8, max_seq_length: int = 6) -> None:
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.tokenizer = _DummyTokenizer()
        self.encode_calls = 0

    def encode(
        self,
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.encode_calls += 1
        rng = np.random.default_rng(42)
        vecs = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)
        return vecs


class TestLexicalAndSemanticPreprocessing(unittest.TestCase):
    def test_inverted_indexer_builds_tf_df_lengths_and_stats(self) -> None:
        indexer = InvertedIndexer(remove_stopwords=True)
        payload = indexer.build(
            {
                "DOC-1": "Vitamin C prevents scurvy and vitamin C helps immunity",
                "DOC-2": "Dietary fiber lowers cholesterol",
            }
        )

        self.assertEqual(payload["N"], 2)
        self.assertGreater(payload["avg_doc_length"], 0)
        self.assertEqual(payload["index"]["vitamin"], {"DOC-1": 2})
        self.assertEqual(payload["df"]["vitamin"], 1)
        self.assertNotIn("and", payload["index"])
        self.assertEqual(payload["vocab_size"], len(payload["index"]))
        self.assertEqual(payload["total_postings"], sum(payload["df"].values()))
        self.assertIn("params", payload)

    def test_english_stemming_groups_terms_in_the_inverted_index(self) -> None:
        corpus = {"D1": "Connections connected 20 t2", "D2": "Connect and health"}
        indexer = InvertedIndexer(stemming="snowball-english")

        self.assertEqual(indexer.tokenize(corpus["D1"]), ["connect", "connect", "20", "t2"])
        self.assertEqual(indexer.tokenize(corpus["D2"]), ["connect", "health"])
        payload = indexer.build(corpus)

        self.assertEqual(payload["params"]["stemming"], "snowball-english")
        self.assertEqual(payload["index"]["connect"], {"D1": 2, "D2": 1})
        self.assertEqual(payload["df"]["connect"], 2)
        self.assertEqual(payload["doc_lengths"], {"D1": 4, "D2": 2})

    def test_stemming_respects_case_and_filters_stopwords_before_stemming(self) -> None:
        indexer = InvertedIndexer(
            lowercase=False,
            token_pattern=r"[A-Za-z0-9]+",
            stopwords={"connections"},
            stemming="snowball-english",
        )

        self.assertEqual(
            indexer.tokenize("Connected connections connecting 20 t2"),
            ["Connected", "connect", "20", "t2"],
        )

    def test_embedding_indexer_uses_suggested_default_and_truncation_info(self) -> None:
        dummy_model = _DummySentenceTransformer(dim=8, max_seq_length=5)
        indexer = EmbeddingIndexer(model=dummy_model)
        self.assertEqual(indexer.embedding_model_name, DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "sentence-transformers/all-MiniLM-L6-v2")

        embeddings, metadata = indexer.index(
            {
                "D1": "short doc",
                "D2": "this document has many tokens exceeding max length",
            }
        )
        self.assertEqual(embeddings.shape, (2, 8))
        self.assertEqual(embeddings.dtype, np.float32)
        self.assertEqual(metadata["doc_ids"], ["D1", "D2"])
        self.assertEqual(metadata["embedding_dim"], 8)
        self.assertEqual(metadata["max_seq_length"], 5)
        self.assertEqual(metadata["truncated_docs_count"], 1)
        self.assertTrue(metadata["had_truncation"])

    def test_preprocessor_caches_and_skips_indexers_on_cache_hit(self) -> None:
        sample_splits = {
            "train": (
                {
                    "MED-1": {"title": "Vitamin C", "text": "Prevents scurvy effectively."},
                    "MED-2": {"title": "Fiber", "text": "Reduces LDL cholesterol levels."},
                },
                {},
                {},
            ),
            "test": (
                {
                    "MED-1": {"title": "Vitamin C", "text": "Prevents scurvy effectively."},
                    "MED-2": {"title": "Fiber", "text": "Reduces LDL cholesterol levels."},
                },
                {"Q-1": "vitamin c"},
                {},
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            runs_dir = tmp_path / "runs"
            dummy_model = _DummySentenceTransformer(dim=12, max_seq_length=128)

            # 1) First run: cache miss -> runs indexers, creates .json and .npy, logs build+cache cost
            pre_build = PreProcessor(
                {
                    "cache_dir": str(cache_dir),
                    "runs_dir": str(runs_dir),
                    "run_id": "run_build_test",
                    "embedding_model": dummy_model,
                }
            )
            result_build = pre_build.preprocess(sample_splits)

            self.assertEqual(dummy_model.encode_calls, 1)
            self.assertTrue((cache_dir / "lexical_index.json").exists())
            self.assertTrue((cache_dir / "semantic_embeddings.npy").exists())
            self.assertTrue((cache_dir / "semantic_metadata.json").exists())

            self.assertEqual(result_build["lexical"]["N"], 2)
            self.assertIn("scurvy", result_build["lexical"]["index"])
            self.assertEqual(result_build["semantic"]["embeddings"].shape, (2, 12))
            self.assertEqual(result_build["semantic"]["doc_ids"], ["MED-1", "MED-2"])

            metrics_file = runs_dir / "run_build_test" / "index_cache_metrics.json"
            self.assertTrue(metrics_file.exists())
            with open(metrics_file, "r", encoding="utf-8") as f:
                logged_build = json.load(f)
            self.assertFalse(logged_build["lexical"]["cache_hit"])
            self.assertFalse(logged_build["semantic"]["cache_hit"])
            self.assertIsNotNone(logged_build["lexical"]["cache_save_time_seconds"])
            self.assertIsNotNone(logged_build["semantic"]["cache_save_time_seconds"])

            # 2) Second run: cache hit -> MUST NOT execute InvertedIndexer or EmbeddingIndexer
            pre_cached = PreProcessor(
                {
                    "cache_dir": str(cache_dir),
                    "runs_dir": str(runs_dir),
                    "run_id": "run_load_test",
                }
            )
            with (
                patch(
                    "src.pre_processing.pre_processor.InvertedIndexer",
                    side_effect=AssertionError("InvertedIndexer should not run on cache hit"),
                ),
                patch(
                    "src.pre_processing.pre_processor.EmbeddingIndexer",
                    side_effect=AssertionError("EmbeddingIndexer should not run on cache hit"),
                ),
            ):
                result_cached = pre_cached.preprocess(None)

            self.assertEqual(result_cached["lexical"]["N"], 2)
            self.assertTrue(
                np.allclose(
                    result_cached["semantic"]["embeddings"],
                    result_build["semantic"]["embeddings"],
                )
            )

            load_metrics_file = runs_dir / "run_load_test" / "index_cache_metrics.json"
            with open(load_metrics_file, "r", encoding="utf-8") as f:
                logged_load = json.load(f)
            self.assertTrue(logged_load["lexical"]["cache_hit"])
            self.assertTrue(logged_load["semantic"]["cache_hit"])
            self.assertEqual(logged_load["lexical"]["status"], "loaded_from_cache")
            self.assertEqual(logged_load["semantic"]["status"], "loaded_from_cache")
            self.assertIsNotNone(logged_load["lexical"]["cache_load_time_seconds"])
            self.assertIsNotNone(logged_load["semantic"]["cache_load_time_seconds"])

    def test_lexical_caches_follow_mode_and_reuse_semantic_embeddings(self) -> None:
        corpus = {"D1": "Connections connected", "D2": "Connect and health"}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            encoder = _DummySentenceTransformer()
            config = {"cache_dir": tmp / "cache", "runs_dir": tmp / "runs", "embedding_model": encoder}
            default = PreProcessor(config)
            default.preprocess(corpus)
            self.assertEqual(default.lexical_cache_path.name, "lexical_index.json")
            legacy = json.loads(default.lexical_cache_path.read_text())
            del legacy["params"]["stemming"]
            default.lexical_cache_path.write_text(json.dumps(legacy))

            stem_config = {**config, "lexical_stemming": "snowball-english"}
            stemmed = PreProcessor(stem_config)
            built = stemmed.preprocess(corpus)
            self.assertEqual(stemmed.lexical_cache_path.name, "lexical_index_snowball_english.json")
            self.assertEqual(built["lexical"]["index"]["connect"], {"D1": 2, "D2": 1})
            self.assertEqual(built["run_metrics"]["lexical"]["stats"]["params"]["stemming"], "snowball-english")
            self.assertTrue(stemmed.preprocess()["run_metrics"]["lexical"]["cache_hit"])
            restored = default.preprocess()
            self.assertTrue(restored["run_metrics"]["lexical"]["cache_hit"])
            self.assertEqual(restored["lexical"]["params"]["stemming"], "none")

            explicit = PreProcessor({**stem_config, "lexical_cache_file": default.lexical_cache_path.name})
            rebuilt = explicit.preprocess(corpus)
            self.assertFalse(rebuilt["run_metrics"]["lexical"]["cache_hit"])
            self.assertEqual(rebuilt["lexical"]["params"]["stemming"], "snowball-english")
            mismatch = rebuilt["lexical"]
            mismatch["params"]["stemmer_version"] = "old"
            explicit.lexical_cache_path.write_text(json.dumps(mismatch))
            rebuilt_version = explicit.preprocess(corpus)
            self.assertFalse(rebuilt_version["run_metrics"]["lexical"]["cache_hit"])
            self.assertNotEqual(rebuilt_version["lexical"]["params"]["stemmer_version"], "old")
            self.assertEqual(encoder.encode_calls, 1)

    def test_empty_cache_files_trigger_rebuild(self) -> None:
        corpus = {"D1": {"title": "Heart", "text": "Omega 3 fatty acids."}}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir(parents=True)
            (cache_dir / "lexical_index.json").write_text("", encoding="utf-8")
            (cache_dir / "semantic_embeddings.npy").write_bytes(b"")
            (cache_dir / "semantic_metadata.json").write_text("", encoding="utf-8")

            dummy_model = _DummySentenceTransformer(dim=4)
            pre = PreProcessor(
                {
                    "cache_dir": str(cache_dir),
                    "runs_dir": str(tmp_path / "runs"),
                    "run_id": "run_empty_cache",
                    "embedding_model": dummy_model,
                }
            )
            res = pre.preprocess(corpus)
            self.assertEqual(dummy_model.encode_calls, 1)
            self.assertEqual(res["lexical"]["N"], 1)
            self.assertFalse(res["run_metrics"]["lexical"]["cache_hit"])
            self.assertFalse(res["run_metrics"]["semantic"]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
