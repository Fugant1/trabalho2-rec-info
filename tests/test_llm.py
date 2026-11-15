import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.retrieval.llm import (
    DEFAULT_GEMINI_MODEL,
    ContextDocument,
    RAGLLM,
    _extract_message_text,
)
from src.utils.config import Config


class _CapturingFakeChatModel(FakeListChatModel):
    """Fake chat model that records received prompt messages for inspection."""

    received_messages: list[list] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class TestRAGLLM(unittest.TestCase):
    def test_no_rag_condition_a_with_none_and_empty_list(self) -> None:
        fake_llm = _CapturingFakeChatModel(responses=["Sem RAG 1", "Sem RAG 2"])
        rag_llm = RAGLLM(language="pt-BR", llm=fake_llm)

        answer_none = rag_llm.generate("What causes scurvy?", documents=None)
        self.assertEqual(answer_none, "Sem RAG 1")

        meta_empty = rag_llm.generate_with_metadata("What causes scurvy?", documents=[])
        self.assertEqual(meta_empty.answer, "Sem RAG 2")
        self.assertEqual(meta_empty.condition, "A")
        self.assertEqual(meta_empty.context_doc_ids, [])
        self.assertEqual(meta_empty.language, "Portuguese")
        self.assertIn("What causes scurvy?", meta_empty.prompt)
        self.assertNotIn("Context documents:", meta_empty.prompt)

    def test_rag_condition_b_with_beir_dicts_and_top_k_default(self) -> None:
        fake_llm = _CapturingFakeChatModel(
            responses=["A vitamina C previne escorbuto [MED-1]."]
        )
        rag_llm = RAGLLM(language="Portuguese", llm=fake_llm)

        docs = [
            {
                "_id": f"MED-{i}",
                "title": f"Study {i}",
                "text": f"Evidence body {i}.",
                "score": 1.0 / i,
            }
            for i in range(1, 8)
        ]

        result = rag_llm.generate_with_metadata(
            "Does vitamin C prevent scurvy?",
            documents=docs,
        )

        self.assertEqual(result.condition, "B")
        self.assertEqual(
            result.context_doc_ids,
            ["MED-1", "MED-2", "MED-3", "MED-4", "MED-5"],
        )
        self.assertIn("[doc_id: MED-1]", result.prompt)
        self.assertIn("Title: Study 1", result.prompt)
        self.assertIn("Text: Evidence body 1.", result.prompt)
        self.assertNotIn("MED-6", result.prompt)
        self.assertIn("insufficient evidence", result.prompt.lower())
        self.assertEqual(result.answer, "A vitamina C previne escorbuto [MED-1].")

    def test_callable_and_various_document_formats(self) -> None:
        fake_llm = _CapturingFakeChatModel(responses=["Resposta consolidada."])
        rag_llm = RAGLLM(llm=fake_llm, top_k_context=None)

        mixed_docs = [
            Document(
                page_content="Doc 1 content",
                metadata={"doc_id": "PLAIN-1", "title": "Title 1"},
            ),
            ("PLAIN-2", 0.91, "Doc 2 content"),
            ("PLAIN-3", {"title": "Title 3", "text": "Doc 3 content"}),
            ContextDocument(doc_id="PLAIN-4", title="Title 4", text="Doc 4 content"),
            "Plain raw string content",
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
        self.assertIn("Doc 2 content", response.prompt)
        self.assertIn("Title: Title 3", response.prompt)
        self.assertIn("Plain raw string content", response.prompt)

        fake_llm2 = _CapturingFakeChatModel(responses=["Call answer"])
        rag_llm2 = RAGLLM(llm=fake_llm2)
        self.assertEqual(rag_llm2("Simple question?"), "Call answer")

    def test_top_k_override_per_call(self) -> None:
        fake_llm = _CapturingFakeChatModel(responses=["Ok"])
        rag_llm = RAGLLM(llm=fake_llm, top_k_context=5)
        docs = [{"doc_id": f"D-{i}", "text": f"T-{i}"} for i in range(5)]

        meta = rag_llm.generate_with_metadata("Query?", documents=docs, top_k=2)
        self.assertEqual(meta.context_doc_ids, ["D-0", "D-1"])

    def test_gemini_initialization_requires_api_key(self) -> None:
        rag_llm = RAGLLM()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                _ = rag_llm.llm

    def test_gemini_initialization_with_env_key(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-test-key"}, clear=True):
            rag_llm = RAGLLM(
                model=DEFAULT_GEMINI_MODEL,
                temperature=0.0,
                max_output_tokens=512,
                seed=42,
            )
            client = rag_llm.llm
            self.assertIsInstance(client, ChatGoogleGenerativeAI)
            self.assertEqual(client.model, DEFAULT_GEMINI_MODEL)
            self.assertEqual(client.temperature, 0.0)
            self.assertEqual(client.max_output_tokens, 512)
            self.assertEqual(client.seed, 42)

    def test_from_config(self) -> None:
        cfg = Config(
            {
                "llm_model": "gemini-3.6-flash",
                "llm_temperature": 0.0,
                "llm_max_output_tokens": 256,
                "llm_seed": 123,
                "llm_language": "en",
                "rag_top_k": 3,
            }
        )
        rag_llm = RAGLLM.from_config(cfg)
        self.assertEqual(rag_llm.model, "gemini-3.6-flash")
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

    def test_input_validation(self) -> None:
        fake_llm = _CapturingFakeChatModel(responses=["x"])
        rag_llm = RAGLLM(llm=fake_llm)

        with self.assertRaises(ValueError):
            rag_llm.generate("   ")
        with self.assertRaises(TypeError):
            rag_llm.generate("Valid?", documents="not-a-list")
        with self.assertRaises(TypeError):
            rag_llm.generate("Valid?", documents={"doc_id": "1", "text": "a"})
        with self.assertRaises(ValueError):
            RAGLLM(temperature=-0.1)


if __name__ == "__main__":
    unittest.main()
