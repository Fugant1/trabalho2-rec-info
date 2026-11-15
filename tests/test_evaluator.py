import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import Evaluator


class TestEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = Evaluator()

    def test_metrics_match_pytrec_eval_and_linear_ndcg(self) -> None:
        qrels = {
            "Q1": {"D1": 2, "D2": 1, "D3": 0, "D4": 2},
            "Q2": {"D1": 1, "D5": 1},
            "Q3": {"D1": 0},
        }
        rankings = {
            "Q1": [("D2", 0.9), ("D3", 0.8), ("D1", 0.7)],
            "Q2": [{"doc_id": "D5", "score": 0.95}],
            "Q3": {"D1": 0.5},
        }

        res = self.evaluator.evaluate(qrels, rankings, k=10, model_name="bm25")
        trec = self.evaluator.cross_check_with_pytrec_eval(qrels, rankings)

        for qid in qrels:
            pq = res["per_query"][qid]
            self.assertAlmostEqual(pq["P10"], trec[qid]["P_10"], places=12)
            self.assertAlmostEqual(pq["R10"], trec[qid]["recall_10"], places=12)
            self.assertAlmostEqual(pq["AP10"], trec[qid]["map_cut_10"], places=12)
            self.assertAlmostEqual(pq["NDCG10"], trec[qid]["ndcg_cut_10"], places=12)

        mean_ap = sum(res["per_query"][q]["AP10"] for q in qrels) / len(qrels)
        self.assertAlmostEqual(res["global"]["MAP@10"], mean_ap, places=12)

    def test_rag_query_selection_and_retrieval_metrics(self) -> None:
        queries = {f"Q{i}": (f"Question {i}?" if i % 2 == 0 else f"Statement {i}") for i in range(30)}
        test_qrels = {f"Q{i}": {"D1": 2} for i in range(30)}

        selected1 = self.evaluator.select_rag_queries(test_qrels, queries, seed=42, sample_size=10)
        selected2 = self.evaluator.select_rag_queries(test_qrels, queries, seed=42, sample_size=10)
        self.assertEqual(selected1, selected2)
        self.assertEqual(len(selected1), 10)
        self.assertTrue(all(queries[q].endswith("?") for q in selected1))

        rag_qrels = {
            "Q1": {"D1": 2, "D2": 1, "D3": 1, "D4": 1},
            "Q2": {"D9": 1},
        }
        top5 = {
            "Q1": [("D1", 0.9), ("X1", 0.8), ("D2", 0.7), ("X2", 0.6), ("X3", 0.5)],
            "Q2": [("X1", 0.9), ("X2", 0.8), ("X3", 0.7), ("X4", 0.6), ("X5", 0.5)],
        }
        rag_eval = self.evaluator.evaluate_rag_retrieval(rag_qrels, top5)
        self.assertAlmostEqual(rag_eval["per_query"]["Q1"]["P5"], 2 / 5)
        self.assertAlmostEqual(rag_eval["per_query"]["Q1"]["R5"], 2 / 4)
        self.assertEqual(rag_eval["per_query"]["Q1"]["Hit5"], 1.0)
        self.assertEqual(rag_eval["per_query"]["Q2"]["Hit5"], 0.0)
        self.assertAlmostEqual(rag_eval["global"]["Hit@5"], 0.5)

    def test_typo_experiment_recovery_and_validation(self) -> None:
        qrels = {
            "Q1": {"D1": 2, "D2": 1},
            "Q2": {"D1": 2},
        }
        records = [
            {
                "query_id": "Q1",
                "query_original": "vitamin c",
                "query_erro": "vitmin c",
                "query_corrigida": "vitamin c",
                "tipo_erro": "remocao",
                "ranking_original": [("D1", 1.0), ("D2", 0.5)],
                "ranking_erro": [("X1", 1.0), ("D2", 0.5)],
                "ranking_corrigida": [("D1", 1.0), ("D2", 0.5)],
            },
            {
                "query_id": "Q2",
                "query_original": "scurvy",
                "query_erro": "scruvy",
                "query_corrigida": "scurvy",
                "tipo_erro": "troca",
                "NDCG10_original": 0.5,
                "NDCG10_erro": 0.6,
                "NDCG10_corrigida": 0.6,
            },
        ]
        evaluated = self.evaluator.evaluate_typo_experiment(qrels, records)
        self.assertTrue(evaluated[0]["error_reduced_performance"])
        self.assertAlmostEqual(evaluated[0]["recovery_rate"], 1.0)
        self.assertFalse(evaluated[1]["error_reduced_performance"])
        self.assertIsNone(evaluated[1]["recovery_rate"])
        self.assertIn("não reduziu", evaluated[1]["note"])

        with self.assertRaises(ValueError):
            self.evaluator.evaluate_typo_experiment(
                qrels,
                [
                    {
                        "query_id": "Q2",
                        "query_original": "scurvy",
                        "query_erro": "x",
                        "query_corrigida": "y",
                        "tipo_erro": "substituicao",
                        "NDCG10_original": 0.0,
                        "NDCG10_erro": 0.0,
                        "NDCG10_corrigida": 0.0,
                    }
                ],
            )

    def test_compare_lexical_vs_semantic_and_analyze_reranking(self) -> None:
        qrels = {f"Q{i}": {"D1": 2} for i in range(1, 7)}
        lexical = {
            "Q1": [("X1", 1.0)],
            "Q2": [("X1", 1.0)],
            "Q3": [("X1", 1.0)],
            "Q4": [("D1", 1.0)],
            "Q5": [("D1", 1.0)],
            "Q6": [("D1", 1.0)],
        }
        semantic = {
            "Q1": [("D1", 1.0)],
            "Q2": [("D1", 1.0)],
            "Q3": [("D1", 1.0)],
            "Q4": [("X1", 1.0)],
            "Q5": [("X1", 1.0)],
            "Q6": [("X1", 1.0)],
        }
        cmp_res = self.evaluator.compare_lexical_vs_semantic(qrels, lexical, semantic)
        self.assertEqual([c["query_id"] for c in cmp_res["semantic_advantage"]], ["Q1", "Q2", "Q3"])
        self.assertEqual([c["query_id"] for c in cmp_res["lexical_advantage"]], ["Q4", "Q5", "Q6"])

        reranked = {
            "Q1": [("D1", 0.9), ("X1", 0.1)],  # gain
            "Q2": [("D1", 0.9), ("X1", 0.1)],  # gain
            "Q3": [("X1", 0.9)],               # no change (0 -> 0)
            "Q4": [("X1", 0.9), ("D1", 0.1)],  # drop (rank 1 -> rank 2)
            "Q5": [("D1", 0.9)],               # no change (1 -> 1)
            "Q6": [("D1", 0.9)],               # no change (1 -> 1)
        }
        rer_analysis = self.evaluator.analyze_reranking(qrels, lexical, reranked)
        self.assertEqual([c["query_id"] for c in rer_analysis["largest_gains"]], ["Q1", "Q2"])
        self.assertEqual(rer_analysis["worst_drop"]["query_id"], "Q4")
        self.assertAlmostEqual(rer_analysis["minimal_change"]["ndcg_delta"], 0.0)
        self.assertTrue(len(rer_analysis["largest_gains"][0]["moved_docs"]) > 0)

    def test_rag_human_aggregation_and_csv_exports(self) -> None:
        rag_rows = [
            {
                "query_id": "PLAIN-68",
                "query": "Q?",
                "condition": "A",
                "context_doc_ids": [],
                "prompt": "P A",
                "answer": "Ans A",
                "fidelidade": 1,
                "justificativa_fidelidade": "ok",
                "completude": 1,
                "justificativa_completude": "ok",
                "utilidade": 2,
                "justificativa_utilidade": "ok",
                "rastreabilidade": "",
                "justificativa_rastreabilidade": "sem RAG",
            },
            {
                "query_id": "PLAIN-68",
                "query": "Q?",
                "condition": "B",
                "context_doc_ids": ["D1", "D2"],
                "prompt": "P B",
                "answer": "Ans B",
                "fidelidade": 2,
                "justificativa_fidelidade": "ok",
                "completude": 2,
                "justificativa_completude": "ok",
                "utilidade": 2,
                "justificativa_utilidade": "ok",
                "rastreabilidade": 2,
                "justificativa_rastreabilidade": "citou D1",
            },
        ]
        agg = self.evaluator.aggregate_rag_human_scores(rag_rows)
        self.assertEqual(agg["A"]["fidelidade"], 1.0)
        self.assertIsNone(agg["A"]["rastreabilidade"])
        self.assertEqual(agg["B"]["rastreabilidade"], 2.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            qrels = {"Q1": {"D1": 2}}
            rankings = {"Q1": [(f"D{i}", 100.0 - i) for i in range(1, 121)]}

            r_csv = self.evaluator.export_ranking_csv(rankings, qrels, tmp / "results_lexical.csv")
            with open(r_csv, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 100)
            self.assertEqual(rows[0]["relevance"], "2")
            self.assertEqual(rows[1]["relevance"], "0")

            ev = self.evaluator.evaluate(qrels, rankings, model_name="lexical")
            m_csv = self.evaluator.export_metrics_per_query_csv({"lexical": ev}, tmp / "metrics_per_query.csv")
            with open(m_csv, encoding="utf-8") as f:
                m_rows = list(csv.DictReader(f))
            self.assertEqual(list(m_rows[0].keys()), ["query_id", "model", "P10", "R10", "AP10", "NDCG10"])

            t_csv = self.evaluator.export_typo_experiment_csv(
                [
                    {
                        "query_id": "Q1",
                        "query_original": "a",
                        "query_erro": "b",
                        "query_corrigida": "a",
                        "tipo_erro": "substituicao",
                        "NDCG10_original": 1.0,
                        "NDCG10_erro": 0.5,
                        "NDCG10_corrigida": 1.0,
                        "recovery_rate": 1.0,
                        "error_reduced_performance": True,
                        "note": "",
                    }
                ],
                tmp / "typo_experiment.csv",
            )
            self.assertTrue(t_csv.exists())

            rag_csv = self.evaluator.export_rag_experiment_csv(rag_rows, tmp / "rag_experiment.csv")
            with open(rag_csv, encoding="utf-8") as f:
                saved_rag = list(csv.DictReader(f))
            self.assertEqual(saved_rag[0]["rastreabilidade"], "N/A")
            self.assertEqual(saved_rag[1]["context_doc_ids"], "D1;D2")


if __name__ == "__main__":
    unittest.main()
