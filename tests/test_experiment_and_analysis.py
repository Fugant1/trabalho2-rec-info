import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from langchain_core.language_models.fake_chat_models import FakeListChatModel

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import compute_cohens_kappa
from src.experiment import Experiment
from src.main import build_parser


class _TinyDataLoader:
    def __init__(self) -> None:
        self.corpus = {
            "MED-1": {"title": "Vitamin C and Scurvy", "text": "Vitamin C prevents scurvy effectively in humans."},
            "MED-2": {"title": "Soluble Dietary Fiber", "text": "Dietary fiber lowers LDL cholesterol levels significantly."},
            "MED-3": {"title": "Curcumin Inflammation", "text": "Turmeric curcumin reduces joint inflammation in arthritis."},
            "MED-4": {"title": "Omega Fatty Acids", "text": "Flaxseed omega fatty acids protect cardiovascular heart health."},
            "MED-5": {"title": "Blueberries Antioxidants", "text": "Blueberries contain potent antioxidants supporting brain memory."},
            "MED-6": {"title": "Green Tea Catechins", "text": "Green tea catechins improve metabolic insulin sensitivity."},
            "MED-7": {"title": "Garlic Blood Pressure", "text": "Aged garlic extract lowers systolic blood pressure naturally."},
            "MED-8": {"title": "Broccoli Sulforaphane", "text": "Broccoli sprouts provide sulforaphane supporting liver detoxification."},
        }
        self.dev_queries = {
            "DEV-1": "does vitamin c prevent scurvy?",
            "DEV-2": "does dietary fiber lower cholesterol?",
            "DEV-3": "does curcumin reduce inflammation?",
            "DEV-4": "do blueberries provide antioxidants?",
        }
        self.dev_qrels = {
            "DEV-1": {"MED-1": 2},
            "DEV-2": {"MED-2": 2},
            "DEV-3": {"MED-3": 2},
            "DEV-4": {"MED-5": 2},
        }
        self.test_queries = {
            "Q-1": "does vitamin c prevent scurvy?",
            "Q-2": "does dietary fiber lower cholesterol?",
            "Q-3": "does curcumin reduce joint inflammation?",
            "Q-4": "do flaxseed omega acids protect heart health?",
            "Q-5": "do blueberries contain antioxidants for memory?",
            "Q-6": "does green tea improve insulin sensitivity?",
            "Q-7": "does garlic lower blood pressure?",
            "Q-8": "does broccoli provide sulforaphane for detoxification?",
        }
        self.test_qrels = {
            "Q-1": {"MED-1": 2, "MED-5": 1},
            "Q-2": {"MED-2": 2, "MED-4": 1},
            "Q-3": {"MED-3": 2},
            "Q-4": {"MED-4": 2, "MED-2": 1},
            "Q-5": {"MED-5": 2},
            "Q-6": {"MED-6": 2},
            "Q-7": {"MED-7": 2},
            "Q-8": {"MED-8": 2},
        }

    def load_data(self):
        return {
            "train": (self.corpus, self.dev_queries, self.dev_qrels),
            "dev": (self.corpus, self.dev_queries, self.dev_qrels),
            "test": (self.corpus, self.test_queries, self.test_qrels),
        }


class _SyntheticBiEncoder:
    def __init__(self) -> None:
        self.max_seq_length = 64
        self.terms = [
            "vitamin", "scurvy", "fiber", "cholesterol", "curcumin", "inflammation",
            "omega", "heart", "blueberries", "antioxidants", "tea", "insulin",
            "garlic", "pressure", "broccoli", "sulforaphane",
        ]

    def encode(
        self,
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        mat = np.zeros((len(texts), len(self.terms)), dtype=np.float32)
        for i, txt in enumerate(texts):
            low = txt.lower()
            for j, term in enumerate(self.terms):
                if term in low:
                    mat[i, j] = 1.0
            if np.linalg.norm(mat[i]) == 0:
                mat[i, 0] = 0.01
        if normalize_embeddings:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / np.maximum(norms, 1e-12)
        return mat


class _SyntheticCrossEncoder:
    max_length = 64

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        scores = []
        for q, d in pairs:
            qw = set(q.lower().replace("?", "").split())
            dw = set(d.lower().replace(".", "").split())
            scores.append(float(len(qw & dw)))
        return np.asarray(scores, dtype=np.float32)


class TestExperimentEndToEnd(unittest.TestCase):
    def test_full_pipeline_generates_all_required_csvs_and_blind_aggregation(self) -> None:
        # 4 RAG queries x 2 conditions (A and B) = 8 LLM calls
        fake_answers = [
            "Sem RAG resposta.",
            "Com RAG resposta baseada em [MED-1].",
        ] * 4
        fake_llm = FakeListChatModel(responses=fake_answers)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            config = {
                "dataset_name": "synthetic_nfcorpus",
                "cache_dir": str(tmp / "cache"),
                "runs_dir": str(tmp / "runs"),
                "results_dir": str(tmp / "results"),
                "run_id": "test_run",
                "rrf_k_candidates": [10, 60, 100],
                "rerank_top_k": 5,
                "typo_num_queries": 4,
                "typo_tune_on_dev": True,
                "typo_eval_semantic": True,
                "rag_sample_size": 4,
                "rag_top_k": 5,
                "llm_language": "Portuguese",
            }
            exp = Experiment(
                config,
                data_loader=_TinyDataLoader(),
                embedding_model=_SyntheticBiEncoder(),
                reranker_model=_SyntheticCrossEncoder(),
                llm_client=fake_llm,
            )
            summary = exp.run(["all"])

            results_dir = tmp / "results"
            # Verify all 7 mandatory standardized CSVs (Requirement 15)
            required_csvs = [
                "results_lexical.csv",
                "results_semantic.csv",
                "results_hybrid.csv",
                "results_reranking.csv",
                "metrics_per_query.csv",
                "typo_experiment.csv",
                "rag_experiment.csv",
            ]
            for fname in required_csvs:
                path = results_dir / fname
                self.assertTrue(path.is_file(), f"Missing required CSV: {fname}")

            # Verify auxiliary analysis outputs (Reqs 7, 10, 13, 14)
            for aux in (
                "dev_tuning_results.json",
                "analysis_lexical_vs_semantic.json",
                "analysis_reranking.json",
                "rag_blind_evaluation.csv",
                "rag_blind_key.json",
                "rag_context_reference.json",
                "rag_pipeline_trace.json",
                "experiment_summary.json",
            ):
                self.assertTrue((results_dir / aux).is_file(), f"Missing auxiliary file: {aux}")

            # Verify metrics_per_query.csv has all 4 models and mean AP10 == MAP@10
            with open(results_dir / "metrics_per_query.csv", encoding="utf-8") as f:
                m_rows = list(csv.DictReader(f))
            self.assertEqual(
                {r["model"] for r in m_rows},
                {"lexical", "semantic", "hybrid", "reranking"},
            )
            lex_aps = [float(r["AP10"]) for r in m_rows if r["model"] == "lexical"]
            self.assertAlmostEqual(
                sum(lex_aps) / len(lex_aps),
                summary["retrieval"]["global_metrics"]["lexical"]["MAP@10"],
                places=12,
            )

            # Verify typo_experiment.csv schema
            with open(results_dir / "typo_experiment.csv", encoding="utf-8") as f:
                t_rows = list(csv.DictReader(f))
            self.assertEqual(len(t_rows), 4)
            self.assertTrue(all(float(r["NDCG10_original"]) > 0 for r in t_rows))

            # Verify RAG blind evaluation + 2-annotator unblinding & Cohen's Kappa
            blind_template = results_dir / "rag_blind_evaluation.csv"
            with open(blind_template, encoding="utf-8") as f:
                blind_rows = list(csv.DictReader(f))
            self.assertEqual(len(blind_rows), 8)

            ann1_path = results_dir / "blind_ann1.csv"
            ann2_path = results_dir / "blind_ann2.csv"
            for ann_path, offset in ((ann1_path, 0), (ann2_path, 0)):
                filled = []
                for row in blind_rows:
                    r = dict(row)
                    is_cited = "[MED-" in r["answer"]
                    r["fidelidade"] = "2" if is_cited else "1"
                    r["justificativa_fidelidade"] = "verificado no contexto"
                    r["completude"] = "2"
                    r["justificativa_completude"] = "completo"
                    r["utilidade"] = "2"
                    r["justificativa_utilidade"] = "util"
                    r["rastreabilidade"] = "2" if is_cited else "0"
                    r["justificativa_rastreabilidade"] = "citou fonte" if is_cited else "sem fonte"
                    filled.append(r)
                with open(ann_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(filled[0].keys()))
                    writer.writeheader()
                    writer.writerows(filled)

            agg = exp.aggregate_rag_scores([ann1_path, ann2_path])
            self.assertIsNone(agg["condition_means"]["A"]["rastreabilidade"])
            self.assertEqual(agg["condition_means"]["B"]["rastreabilidade"], 2.0)
            self.assertEqual(agg["condition_means"]["B"]["fidelidade"], 2.0)
            self.assertEqual(agg["condition_means"]["A"]["fidelidade"], 1.0)
            self.assertEqual(agg["inter_annotator_agreement"]["fidelidade"]["exact_agreement"], 1.0)
            self.assertAlmostEqual(agg["inter_annotator_agreement"]["fidelidade"]["cohens_kappa"], 1.0)

            # Verify rag_experiment.csv got updated with N/A for A rastreabilidade
            with open(results_dir / "rag_experiment.csv", encoding="utf-8") as f:
                final_rag = list(csv.DictReader(f))
            cond_a_rows = [r for r in final_rag if r["condition"] == "A"]
            self.assertTrue(all(r["rastreabilidade"] == "N/A" for r in cond_a_rows))

            # Verify pipeline trace JSON has all 6 stages
            with open(results_dir / "rag_pipeline_trace.json", encoding="utf-8") as f:
                traces = json.load(f)
            first_trace = next(iter(traces.values()))
            for stage_key in (
                "stage_1_lexical",
                "stage_2_semantic",
                "stage_3_hybrid",
                "stage_4_reranking",
                "stage_5_context_selection",
                "stage_6_generation",
            ):
                self.assertIn(stage_key, first_trace)

    def test_cohens_kappa_partial_agreement_and_cli_parser(self) -> None:
        kappa = compute_cohens_kappa([0, 1, 2, 2, 1, 0], [0, 1, 2, 1, 1, 0])
        self.assertIsNotNone(kappa)
        assert kappa is not None
        self.assertGreater(kappa, 0.5)
        self.assertLess(kappa, 1.0)

        parser = build_parser()
        args = parser.parse_args(
            ["--stages", "setup", "dev-tuning", "retrieval", "typo", "--tune-bm25"]
        )
        self.assertEqual(args.stages, ["setup", "dev-tuning", "retrieval", "typo"])
        self.assertTrue(args.tune_bm25_on_dev)
        self.assertEqual(args.rrf_k_candidates, [10, 60, 100])


if __name__ == "__main__":
    unittest.main()
