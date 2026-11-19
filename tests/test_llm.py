import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.llm import (
    DEFAULT_QWEN_MODEL,
    ContextDocument,
    QuantizedLocalChatModel,
    RAGLLM,
    _extract_message_text,
)
from src.utils.config import Config


class TestRAGLLM(unittest.TestCase):
    rag_llm: RAGLLM
    shared_llm: QuantizedLocalChatModel

    @classmethod
    def setUpClass(cls) -> None:
        # Load Qwen3-4B quantized to 4-bit NF4 once and reuse across all tests
        # to stay well within 16 GB RAM and 6 GB VRAM.
        cls.rag_llm = RAGLLM(
            model="qwen3:4b",
            temperature=0.0,
            max_output_tokens=160,
            seed=42,
            language="pt-BR",
            load_in_4bit=True,
            enable_thinking=False,
        )
        client = cls.rag_llm.llm
        assert isinstance(client, QuantizedLocalChatModel)
        cls.shared_llm = client

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "shared_llm") and cls.shared_llm is not None:
            cls.shared_llm.unload()

    def test_qwen3_4b_loaded_in_4bit(self) -> None:
        self.assertEqual(self.rag_llm.model, DEFAULT_QWEN_MODEL)
        self.assertEqual(self.rag_llm._resolved_provider(), "local")
        self.assertIsInstance(self.shared_llm, QuantizedLocalChatModel)
        self.assertEqual(self.shared_llm.model_name, DEFAULT_QWEN_MODEL)
        self.assertTrue(self.shared_llm.is_loaded_in_4bit)
        self.assertIsNotNone(self.shared_llm.quantization_config)
        self.assertTrue(self.shared_llm.quantization_config.load_in_4bit)
        self.assertEqual(self.shared_llm.quantization_config.bnb_4bit_quant_type, "nf4")
        self.assertTrue(self.shared_llm.quantization_config.bnb_4bit_use_double_quant)
        self.assertEqual(self.shared_llm.temperature, 0.0)
        self.assertEqual(self.shared_llm.seed, 42)

    def test_no_rag_condition_a_with_none_and_empty_list(self) -> None:
        answer_none = self.rag_llm.generate("What causes scurvy?", documents=None)
        self.assertIsInstance(answer_none, str)
        self.assertTrue(answer_none.strip())

        meta_empty = self.rag_llm.generate_with_metadata(
            "What causes scurvy?",
            documents=[],
        )
        self.assertTrue(meta_empty.answer.strip())
        self.assertEqual(meta_empty.answer, answer_none)
        self.assertEqual(meta_empty.condition, "A")
        self.assertEqual(meta_empty.context_doc_ids, [])
        self.assertEqual(meta_empty.model, DEFAULT_QWEN_MODEL)
        self.assertEqual(meta_empty.language, "Portuguese")
        self.assertIn("What causes scurvy?", meta_empty.prompt)
        self.assertNotIn("Context documents:", meta_empty.prompt)

    def test_rag_condition_b_with_beir_dicts_and_top_k_default(self) -> None:
        docs = [
            {
                "_id": "MED-1",
                "title": "Vitamin C and Scurvy Prevention",
                "text": (
                    "Clinical studies confirm that adequate vitamin C "
                    "(ascorbic acid) intake prevents and reverses scurvy in humans."
                ),
                "score": 1.0,
            },
            *[
                {
                    "_id": f"MED-{i}",
                    "title": f"Nutrition Study {i}",
                    "text": f"General dietary observation {i}.",
                    "score": 1.0 / i,
                }
                for i in range(2, 8)
            ],
        ]

        result = self.rag_llm.generate_with_metadata(
            "Does vitamin C prevent scurvy?",
            documents=docs,
        )

        self.assertEqual(result.condition, "B")
        self.assertEqual(result.model, DEFAULT_QWEN_MODEL)
        self.assertEqual(
            result.context_doc_ids,
            ["MED-1", "MED-2", "MED-3", "MED-4", "MED-5"],
        )
        self.assertIn("[doc_id: MED-1]", result.prompt)
        self.assertIn("Title: Vitamin C and Scurvy Prevention", result.prompt)
        self.assertIn("prevents and reverses scurvy", result.prompt)
        self.assertNotIn("MED-6", result.prompt)
        self.assertIn("insufficient evidence", result.prompt.lower())
        self.assertTrue(result.answer.strip())
        self.assertIn("MED-1", result.answer)

    def test_callable_and_various_document_formats(self) -> None:
        rag_llm = RAGLLM(
            model="qwen3:4b",
            temperature=0.0,
            max_output_tokens=160,
            seed=42,
            top_k_context=None,
            load_in_4bit=True,
            llm=self.shared_llm,
        )

        mixed_docs = [
            Document(
                page_content="Soluble fiber lowers LDL cholesterol levels.",
                metadata={"doc_id": "PLAIN-1", "title": "Title 1"},
            ),
            ("PLAIN-2", 0.91, "Oat beta-glucan helps reduce blood cholesterol."),
            ("PLAIN-3", {"title": "Title 3", "text": "Whole grains provide fiber."}),
            ContextDocument(
                doc_id="PLAIN-4",
                title="Title 4",
                text="High fiber diets support cardiovascular health.",
            ),
            "Plain raw string content about dietary fiber.",
        ]

        response = rag_llm.generate_with_metadata(
            "How does fiber affect cholesterol?",
            documents=mixed_docs,
        )
        self.assertEqual(
            response.context_doc_ids,
            ["PLAIN-1", "PLAIN-2", "PLAIN-3", "PLAIN-4", "DOC-5"],
        )
        self.assertIn("Title: Title 1", response.prompt)
        self.assertIn("Oat beta-glucan", response.prompt)
        self.assertIn("Title: Title 3", response.prompt)
        self.assertIn("Plain raw string content", response.prompt)
        self.assertTrue(response.answer.strip())

        call_answer = rag_llm("What is vitamin C?")
        self.assertIsInstance(call_answer, str)
        self.assertTrue(call_answer.strip())

    def test_top_k_override_per_call(self) -> None:
        docs = [
            {"doc_id": "D-0", "text": "Vitamin C prevents scurvy."},
            {"doc_id": "D-1", "text": "Citrus fruits contain vitamin C."},
            {"doc_id": "D-2", "text": "Unrelated topic 2."},
            {"doc_id": "D-3", "text": "Unrelated topic 3."},
            {"doc_id": "D-4", "text": "Unrelated topic 4."},
        ]

        meta = self.rag_llm.generate_with_metadata(
            "What prevents scurvy?",
            documents=docs,
            top_k=2,
        )
        self.assertEqual(meta.context_doc_ids, ["D-0", "D-1"])
        self.assertNotIn("D-2", meta.prompt)
        self.assertTrue(meta.answer.strip())

    def test_from_config(self) -> None:
        cfg = Config(
            {
                "llm_model": "qwen3:4b",
                "llm_temperature": 0.0,
                "llm_max_output_tokens": 256,
                "llm_seed": 123,
                "llm_language": "en",
                "llm_load_in_4bit": True,
                "rag_top_k": 3,
            }
        )
        rag_llm = RAGLLM.from_config(cfg)
        self.assertEqual(rag_llm.model, DEFAULT_QWEN_MODEL)
        self.assertEqual(rag_llm._resolved_provider(), "local")
        self.assertTrue(rag_llm.load_in_4bit)
        self.assertEqual(rag_llm.temperature, 0.0)
        self.assertEqual(rag_llm.max_output_tokens, 256)
        self.assertEqual(rag_llm.seed, 123)
        self.assertEqual(rag_llm.language, "English")
        self.assertEqual(rag_llm.top_k_context, 3)

    def test_extract_message_text_with_thinking_blocks(self) -> None:
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "internal trace"},
                {"type": "text", "text": "Final grounded answer [PLAIN-68]."},
            ]
        )
        self.assertEqual(
            _extract_message_text(msg),
            "Final grounded answer [PLAIN-68].",
        )

        qwen_think_msg = AIMessage(
            content="<think>step by step reasoning</think>\nGrounded answer [MED-1]."
        )
        self.assertEqual(
            _extract_message_text(qwen_think_msg),
            "Grounded answer [MED-1].",
        )

    def test_input_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.rag_llm.generate("   ")
        with self.assertRaises(TypeError):
            self.rag_llm.generate("Valid?", documents="not-a-list")
        with self.assertRaises(TypeError):
            self.rag_llm.generate("Valid?", documents={"doc_id": "1", "text": "a"})
        with self.assertRaises(ValueError):
            RAGLLM(temperature=-0.1)


if __name__ == "__main__":
    unittest.main()
