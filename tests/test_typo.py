import random
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import Evaluator
from src.pre_processing import (
    InvertedIndexer,
    SpellCorrector,
    TypoHandler,
    build_typo_benchmark,
    damerau_levenshtein_distance,
    extract_kgrams,
    introduce_typo,
    jaccard_coefficient,
    levenshtein_distance,
    tune_spell_corrector_on_dev,
)
from src.retrieval import BM25Retriever


class TestTypoTolerance(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = {
            "D1": {"title": "Vitamin C", "text": "Ascorbic acid prevents scurvy and boosts immunity."},
            "D2": {"title": "Dietary Fiber", "text": "Soluble fiber reduces cholesterol and cardiovascular risk."},
            "D3": {"title": "Turmeric Curcumin", "text": "Curcumin reduces chronic inflammation in joints."},
            "D4": {"title": "Omega Fatty Acids", "text": "Flaxseed provides omega fatty acids for heart health."},
            "D5": {"title": "Antioxidant Berries", "text": "Blueberries contain strong antioxidants for brain function."},
        }
        self.corrector = SpellCorrector(k=2, jaccard_threshold=0.25, max_edit_distance=2)
        self.corrector.fit_from_corpus(self.corpus)

    def test_kgrams_jaccard_and_edit_distances(self) -> None:
        grams = extract_kgrams("scurvy", k=2, pad=True)
        self.assertEqual(grams, {"$s", "sc", "cu", "ur", "rv", "vy", "y$"})
        self.assertAlmostEqual(jaccard_coefficient(grams, grams), 1.0)
        self.assertEqual(jaccard_coefficient(set(), set()), 0.0)

        # Insertion, deletion, substitution = 1 in both
        self.assertEqual(levenshtein_distance("vitamin", "vitmin"), 1)
        self.assertEqual(levenshtein_distance("vitamin", "vitaminn"), 1)
        self.assertEqual(levenshtein_distance("vitamin", "vitamun"), 1)
        # Adjacent transposition: 2 in Levenshtein, 1 in Damerau-Levenshtein
        self.assertEqual(levenshtein_distance("scurvy", "scruvy"), 2)
        self.assertEqual(damerau_levenshtein_distance("scurvy", "scruvy"), 1)

    def test_dictionary_preserves_stopwords_and_corrects_all_four_error_types(self) -> None:
        # Stopwords like "what", "and", "the" must be known and never miscorrected
        self.assertTrue(self.corrector.is_known("what"))
        self.assertTrue(self.corrector.is_known("and"))
        self.assertTrue(self.corrector.is_known("cholesterol"))

        cases = {
            "insercao": ("what vitamiin prevents scurvy?", "what vitamin prevents scurvy?"),
            "remocao": ("soluble fiber reduces cholesterl", "soluble fiber reduces cholesterol"),
            "substituicao": ("curcumin reduces inflammatoin?", "curcumin reduces inflammation?"),
            "troca": ("what prevents scruvy?", "what prevents scurvy?"),
        }
        for err_type, (typo_q, expected_q) in cases.items():
            corrected = self.corrector.correct(typo_q)
            self.assertEqual(corrected, expected_q, f"Failed for {err_type}")

        handler = TypoHandler(self.corrector)
        self.assertEqual(handler.correct_typos("vitmin prevents scruvy"), "vitamin prevents scurvy")

    def test_validate_typo_pair_rejects_in_vocabulary_word(self) -> None:
        valid = self.corrector.validate_typo_pair("vitamin prevents scurvy", "vitmin prevents scurvy")
        self.assertTrue(valid["is_valid"])
        self.assertEqual(valid["oov_tokens"], ["vitmin"])

        # Replacing one valid word with another valid word is NOT detectable (Req 4-(ii))
        invalid = self.corrector.validate_typo_pair("vitamin prevents scurvy", "fiber prevents scurvy")
        self.assertFalse(invalid["is_valid"])
        self.assertEqual(invalid["oov_tokens"], [])
        self.assertEqual(invalid["accidental_valid_words"], ["fiber"])

    def test_introduce_typo_and_benchmark_builder_and_dev_tuning(self) -> None:
        rng = random.Random(42)
        for err_type in ("insercao", "remocao", "substituicao", "troca"):
            corrupted = introduce_typo("cholesterol", err_type, self.corrector.vocabulary, rng)
            self.assertIsNotNone(corrupted)
            assert corrupted is not None
            self.assertNotIn(corrupted, self.corrector.vocabulary)
            self.assertNotEqual(corrupted, "cholesterol")

        queries = {
            "Q1": "vitamin prevents scurvy",
            "Q2": "soluble fiber reduces cholesterol",
            "Q3": "curcumin reduces chronic inflammation",
            "Q4": "blueberries strong antioxidants brain",
        }
        benchmark = build_typo_benchmark(
            queries,
            list(queries.keys()),
            self.corrector.vocabulary,
            num_queries=4,
            seed=42,
        )
        self.assertEqual(len(benchmark), 4)
        self.assertEqual(
            {b["tipo_erro"] for b in benchmark},
            {"insercao", "remocao", "substituicao", "troca"},
        )
        for case in benchmark:
            self.assertTrue(
                self.corrector.validate_typo_pair(
                    case["query_original"], case["query_erro"]
                )["is_valid"]
            )

        # Dev-set hyperparameter tuning
        flat_corpus = {
            doc_id: f"{d['title']} {d['text']}" for doc_id, d in self.corpus.items()
        }
        lex_index = InvertedIndexer().build(flat_corpus)
        bm25 = BM25Retriever(lex_index)
        dev_qrels = {
            "Q1": {"D1": 2},
            "Q2": {"D2": 2},
            "Q3": {"D3": 2},
            "Q4": {"D5": 2},
        }
        tuning = tune_spell_corrector_on_dev(
            corpus_documents=self.corpus,
            dev_queries=queries,
            dev_qrels=dev_qrels,
            lexical_retriever=bm25,
            evaluator=Evaluator(),
            k_values=(2, 3),
            jaccard_thresholds=(0.2, 0.35),
            max_edit_distances=(2,),
            num_dev_queries=4,
            seed=42,
        )
        self.assertIn("best_params", tuning)
        self.assertGreaterEqual(tuning["best_result"]["mean_ndcg10_corrigida"], 0.99)


if __name__ == "__main__":
    unittest.main()
