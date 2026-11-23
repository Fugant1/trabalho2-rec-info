import sys
import unittest
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import Evaluator
from src.pre_processing import InvertedIndexer
from src.retrieval import (
    BM25Retriever,
    CrossEncoderReranker,
    HybridRetriever,
    Retriever,
    SemanticRetriever,
    reciprocal_rank_fusion,
)


class _DeterministicBiEncoder:
    """Tiny keyword-sensitive dummy bi-encoder for deterministic unit tests."""

    def __init__(self) -> None:
        self.vocab = ["vitamin", "scurvy", "fiber", "cholesterol", "curcumin", "heart"]

    def encode(
        self,
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        vecs = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(texts):
            low = text.lower()
            for j, word in enumerate(self.vocab):
                if word in low:
                    vecs[i, j] = 1.0
            if np.linalg.norm(vecs[i]) == 0:
                vecs[i, 0] = 0.01
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)
        return vecs


class _DummyTokenizer:
    def __call__(self, queries, docs, truncation=False, add_special_tokens=True, verbose=False):
        return {
            "input_ids": [
                [1] * (len(q.split()) + len(d.split()) + 3)
                for q, d in zip(queries, docs)
            ]
        }


class _DummyCrossEncoder:
    def __init__(self, max_length: int = 8) -> None:
        self.max_length = max_length
        self.tokenizer = _DummyTokenizer()

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        # Return negative logits to verify monotonic tail adjustment with pytrec_eval
        scores = []
        for query, doc in pairs:
            q_words = set(query.lower().split())
            d_words = set(doc.lower().split())
            overlap = len(q_words & d_words)
            scores.append(-5.0 + 2.0 * overlap)
        return np.asarray(scores, dtype=np.float32)


class _StrictDocLengths(dict):
    """Dict that raises if someone tries to iterate over all docs during BM25 search."""

    def __iter__(self):
        raise AssertionError("BM25 must NOT iterate over all documents in the corpus!")

    def items(self):
        raise AssertionError("BM25 must NOT iterate over all documents in the corpus!")

    def keys(self):
        raise AssertionError("BM25 must NOT iterate over all documents in the corpus!")


class TestRetrievalModules(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = {
            "D1": {"title": "Vitamin C", "text": "Vitamin C prevents scurvy effectively."},
            "D2": {"title": "Dietary Fiber", "text": "Dietary fiber reduces cholesterol levels."},
            "D3": {"title": "Curcumin", "text": "Curcumin helps chronic joint inflammation."},
            "D4": {"title": "Heart Health", "text": "Fiber and exercise support heart health."},
            "D5": {"title": "Scurvy History", "text": "Sailors prevented scurvy with citrus."},
        }
        self.flat_corpus = {
            doc_id: f"{d['title']} {d['text']}" for doc_id, d in self.corpus.items()
        }
        self.lexical_index = InvertedIndexer(remove_stopwords=True).build(self.flat_corpus)
        self.bi_encoder = _DeterministicBiEncoder()
        self.doc_ids = list(self.flat_corpus.keys())
        self.doc_embeddings = self.bi_encoder.encode(
            [self.flat_corpus[d] for d in self.doc_ids]
        )

    def test_bm25_uses_posting_lists_only_and_ranks_correctly(self) -> None:
        bm25 = BM25Retriever(self.lexical_index, k1=1.5, b=0.75)
        # Guard doc_lengths against full-corpus scans during search
        bm25.doc_lengths = _StrictDocLengths(bm25.doc_lengths)

        ranked = bm25.search("vitamin scurvy", top_k=5)
        self.assertEqual(ranked[0][0], "D1")
        # Only D1 and D5 contain 'vitamin' or 'scurvy'; non-matching docs are untouched
        self.assertEqual({doc_id for doc_id, _ in ranked}, {"D1", "D5"})

        # Batch retrieve records timing stats
        batch = bm25.retrieve({"Q1": "vitamin scurvy", "Q2": "cholesterol fiber"}, top_k=5)
        self.assertEqual(batch["Q1"][0][0], "D1")
        self.assertEqual(batch["Q2"][0][0], "D2")
        self.assertEqual(bm25.last_timing_stats["num_queries"], 2)
        self.assertGreaterEqual(bm25.last_timing_stats["mean_query_time_seconds"], 0.0)

        # Match explanation provides concrete evidence for Req 7
        evidence = bm25.explain_doc_match("vitamin scurvy unknownterm", "D1")
        self.assertEqual(evidence["missing_terms"], ["unknownterm"])
        self.assertAlmostEqual(evidence["lexical_overlap_ratio"], 2 / 3)
        self.assertGreater(evidence["bm25_score"], 0.0)

    def test_bm25_applies_the_stemming_mode_stored_in_the_index(self) -> None:
        corpus = {"D1": "Connections connected 20 t2", "D2": "Connect and health"}
        original = InvertedIndexer().build(corpus)
        self.assertEqual(original["params"]["stemming"], "none")
        legacy = {**original, "params": original["params"].copy()}
        del legacy["params"]["stemming"]
        for index in (original, legacy):
            bm25 = BM25Retriever(index)
            self.assertEqual(bm25.search("connecting"), [])
            self.assertEqual(bm25.tokenize_query("connecting"), ["connecting"])
            ranked = bm25.search("connect")
            self.assertEqual([doc_id for doc_id, _ in ranked], ["D2"])
            self.assertAlmostEqual(ranked[0][1], 0.8154672712469945)

        stemmed = InvertedIndexer(stemming="snowball-english").build(corpus)
        bm25 = BM25Retriever(stemmed)
        self.assertEqual(bm25.tokenize_query("connecting"), ["connect"])
        metadata = bm25.get_metadata()["preprocessing"]
        self.assertEqual(metadata["stemming"], "snowball-english")
        self.assertEqual(metadata["stemmer_language"], "english")
        self.assertEqual(metadata["stemmer_version"], stemmed["params"]["stemmer_version"])
        ranked = dict(bm25.search("connecting"))
        self.assertEqual(set(ranked), {"D1", "D2"})

        evidence = bm25.explain_doc_match("connecting", "D1")
        self.assertEqual(evidence["query_tokens"], ["connect"])
        self.assertEqual(evidence["matched_terms"][0]["tf"], 2)
        self.assertEqual(evidence["matched_terms"][0]["df"], 2)
        self.assertAlmostEqual(evidence["bm25_score"], ranked["D1"])

    def test_semantic_retriever_cosine_similarity_and_metadata(self) -> None:
        sem = SemanticRetriever(
            embeddings=self.doc_embeddings,
            doc_ids=self.doc_ids,
            metadata={"max_seq_length": 256, "truncated_docs_count": 0, "had_truncation": False},
            model=self.bi_encoder,
        )
        ranked = sem.search("fiber cholesterol", top_k=3)
        self.assertEqual(ranked[0][0], "D2")
        self.assertAlmostEqual(ranked[0][1], 1.0, places=5)

        batch = sem.retrieve({"Q1": "vitamin scurvy", "Q2": "fiber cholesterol"}, top_k=5)
        self.assertEqual(batch["Q1"][0][0], "D1")
        self.assertEqual(batch["Q2"][0][0], "D2")
        meta = sem.get_metadata()
        self.assertEqual(meta["similarity_function"], "cosine")
        self.assertEqual(meta["embedding_dim"], 6)
        self.assertEqual(meta["timing"]["num_queries"], 2)

    def test_hybrid_rrf_formula_fusion_depth_and_dev_tuning(self) -> None:
        lex = [("D1", 10.0), ("D2", 8.0), ("D3", 5.0)]
        sem = [("D2", 0.9), ("D4", 0.8), ("D1", 0.7)]

        fused = reciprocal_rank_fusion([lex, sem], k=60, fusion_depth=2)
        # Depth 2 -> lex takes [D1(1), D2(2)], sem takes [D2(1), D4(2)]
        # D2: 1/62 + 1/61; D1: 1/61; D4: 1/62; D3 excluded
        score_map = dict(fused)
        self.assertNotIn("D3", score_map)
        self.assertAlmostEqual(score_map["D2"], 1 / 62 + 1 / 61, places=12)
        self.assertAlmostEqual(score_map["D1"], 1 / 61, places=12)
        self.assertAlmostEqual(score_map["D4"], 1 / 62, places=12)
        self.assertEqual(fused[0][0], "D2")

        # Dev tuning selects k maximizing dev NDCG@10
        dev_lex = {"Q1": [("D1", 5.0), ("D2", 4.0), ("D3", 3.0)]}
        dev_sem = {"Q1": [("D2", 0.9), ("D3", 0.8), ("D4", 0.7)]}
        dev_qrels = {"Q1": {"D2": 2, "D1": 1}}

        hybrid = HybridRetriever(k=60, fusion_depth=100)
        tuning = hybrid.tune_k(
            dev_lex,
            dev_sem,
            dev_qrels,
            Evaluator(),
            k_candidates=(10, 60, 100),
        )
        self.assertIn(tuning["best_k"], (10, 60, 100))
        self.assertEqual(hybrid.k, tuning["best_k"])
        self.assertEqual(set(tuning["results_by_k"].keys()), {"10", "60", "100"})

    def test_cross_encoder_reranker_reorders_head_preserves_tail_and_tracks_truncation(self) -> None:
        reranker = CrossEncoderReranker(
            rerank_top_k=3,
            batch_size=2,
            model=_DummyCrossEncoder(max_length=8),
        )
        # Initial hybrid ranking has 5 docs: head=[D3, D2, D1], tail=[D4, D5]
        initial = {
            "Q1": [("D3", 0.032), ("D2", 0.031), ("D1", 0.030), ("D4", 0.015), ("D5", 0.010)]
        }
        queries = {"Q1": "vitamin scurvy prevents"}

        reranked = reranker.rerank(queries, initial, self.corpus, track_truncation=True)
        docs_after = [d for d, _ in reranked["Q1"]]
        scores_after = [s for _, s in reranked["Q1"]]

        # D1 matches all 3 query words so it moves to rank 1 within head (top 3)
        self.assertEqual(docs_after[0], "D1")
        self.assertEqual(set(docs_after[:3]), {"D1", "D2", "D3"})
        # Tail below position 3 MUST remain in exact hybrid relative order [D4, D5]
        self.assertEqual(docs_after[3:], ["D4", "D5"])
        # Scores must stay strictly monotonic so pytrec_eval agrees with list order
        self.assertTrue(all(scores_after[i] >= scores_after[i + 1] for i in range(len(scores_after) - 1)))
        self.assertGreater(min(scores_after[:3]), max(scores_after[3:]))

        # Cross-check with pytrec_eval
        qrels = {"Q1": {"D1": 2, "D5": 1}}
        ev = Evaluator()
        ours = ev.evaluate(qrels, reranked, k=10)
        trec = ev.cross_check_with_pytrec_eval(qrels, reranked)
        self.assertAlmostEqual(
            ours["per_query"]["Q1"]["NDCG10"],
            trec["Q1"]["ndcg_cut_10"],
            places=12,
        )

        # Truncation tracking recorded
        meta = reranker.get_metadata()
        self.assertEqual(meta["truncation"]["max_length"], 8)
        self.assertEqual(meta["truncation"]["total_pairs_evaluated"], 3)
        self.assertTrue(meta["truncation"]["had_truncation"])

    def test_unified_retriever_facade(self) -> None:
        bm25 = BM25Retriever(self.lexical_index)
        sem = SemanticRetriever(self.doc_embeddings, self.doc_ids, model=self.bi_encoder)
        hyb = HybridRetriever(k=60, fusion_depth=100)
        rer = CrossEncoderReranker(rerank_top_k=3, model=_DummyCrossEncoder())

        retriever = Retriever(bm25, sem, hyb, rer, corpus=self.corpus)
        all_stages = retriever.retrieve_all_stages({"Q1": "vitamin scurvy"})
        self.assertEqual(set(all_stages.keys()), {"lexical", "semantic", "hybrid", "reranking"})
        self.assertEqual(all_stages["reranking"]["Q1"][0][0], "D1")

        single = retriever.retrieve("vitamin scurvy", top_k=2, mode="reranking")
        self.assertEqual(len(single), 2)
        self.assertEqual(single[0][0], "D1")


if __name__ == "__main__":
    unittest.main()
