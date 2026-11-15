"""LLM generation module using LangChain and Google Gemini (free-tier compatible).

Implements the generation stage of the RAG pipeline defined in the project
specification (Items 11-15):
    query -> hybrid retrieval -> reranking -> context selection -> LLM -> answer

Supports both experimental conditions with the same model and parameters:
    - Condition A (without RAG): ``documents`` is ``None`` or empty; the LLM
      receives only the user question.
    - Condition B (with RAG): ``documents`` provides retrieved context units
      (e.g., Top-5 reranked NFCorpus documents with their ``doc_id``s); the LLM
      is instructed to ground its answer strictly in the context, cite ``doc_id``
      sources, and report when evidence is insufficient.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# Gemini Flash-Lite offers a free tier via Google AI Studio and supports
# deterministic decoding parameters (temperature=0.0, seed).
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

_UNSET = object()

_LANGUAGE_ALIASES = {
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt_br": "Portuguese",
    "portuguese": "Portuguese",
    "português": "Portuguese",
    "en": "English",
    "en-us": "English",
    "en_us": "English",
    "english": "English",
    "inglês": "English",
}

NO_RAG_SYSTEM_PROMPT = (
    "You are a knowledgeable scientific and medical information assistant. "
    "Answer the user's question clearly, accurately, and concisely in {language}. "
    "If you are uncertain or do not have enough knowledge to answer reliably, "
    "explicitly state that."
)

NO_RAG_HUMAN_TEMPLATE = "Question:\n{query}"

RAG_SYSTEM_PROMPT = (
    "You are a scientific and medical information assistant operating in a "
    "Retrieval-Augmented Generation (RAG) pipeline.\n"
    "Follow these rules strictly:\n"
    "1. Answer in {language}.\n"
    "2. Base your answer EXCLUSIVELY on the provided context documents. "
    "Do not use external knowledge or make unsupported extrapolations.\n"
    "3. Cite the exact document identifier(s) in square brackets (e.g., [MED-123]) "
    "for every claim, and include a brief 'Sources' / 'Fontes' list of the cited "
    "doc_ids at the end of your answer.\n"
    "4. If the provided context does not contain sufficient evidence to answer the "
    "question (or only partially addresses it), explicitly state that there is "
    "insufficient evidence in the provided documents and explain what is missing."
)

RAG_HUMAN_TEMPLATE = (
    "Context documents:\n"
    "{context}\n\n"
    "Question:\n"
    "{query}"
)


@dataclass(frozen=True, slots=True)
class ContextDocument:
    """Normalized representation of a retrieved document passed to the LLM."""

    doc_id: str
    text: str
    title: str | None = None
    score: float | None = None

    def formatted_content(self) -> str:
        """Return title and body concatenated, following the BEIR/NFCorpus rule."""
        clean_title = (self.title or "").strip()
        clean_text = self.text.strip()
        if clean_title and clean_text:
            return f"Title: {clean_title}\nText: {clean_text}"
        if clean_title:
            return f"Title: {clean_title}"
        return clean_text


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured output returned by :meth:`RAGLLM.generate_with_metadata`."""

    query: str
    answer: str
    condition: Literal["A", "B"]
    context_doc_ids: list[str]
    prompt: str
    model: str
    temperature: float
    max_output_tokens: int | None
    seed: int | None
    language: str
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize response fields for experiment logging (e.g., rag_experiment.csv)."""
        return {
            "query": self.query,
            "condition": self.condition,
            "context_doc_ids": self.context_doc_ids,
            "prompt": self.prompt,
            "answer": self.answer,
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
            "language": self.language,
            "latency_seconds": self.latency_seconds,
        }


def _normalize_language(language: str) -> str:
    cleaned = language.strip()
    if not cleaned:
        raise ValueError("language must be a non-empty string.")
    return _LANGUAGE_ALIASES.get(cleaned.lower(), cleaned)


def _extract_doc_fields(item: Any, index: int) -> ContextDocument:
    """Convert any supported document representation into a ContextDocument."""
    default_id = f"DOC-{index + 1}"

    if isinstance(item, ContextDocument):
        return item

    if isinstance(item, Document):
        metadata = item.metadata or {}
        doc_id = (
            metadata.get("doc_id")
            or metadata.get("_id")
            or metadata.get("id")
            or item.id
            or default_id
        )
        title = metadata.get("title")
        score = metadata.get("score")
        return ContextDocument(
            doc_id=str(doc_id),
            text=str(item.page_content),
            title=str(title) if title is not None else None,
            score=float(score) if score is not None else None,
        )

    if isinstance(item, Mapping):
        doc_id = (
            item.get("doc_id")
            or item.get("_id")
            or item.get("id")
            or default_id
        )
        raw_text = (
            item.get("text")
            if item.get("text") is not None
            else item.get("content", item.get("page_content", ""))
        )
        title = item.get("title")
        score = item.get("score")
        return ContextDocument(
            doc_id=str(doc_id),
            text=str(raw_text),
            title=str(title) if title is not None else None,
            score=float(score) if score is not None else None,
        )

    if isinstance(item, tuple):
        if len(item) == 2:
            doc_id, payload = item
            if isinstance(payload, Mapping):
                merged = dict(payload)
                merged.setdefault("doc_id", doc_id)
                return _extract_doc_fields(merged, index)
            return ContextDocument(doc_id=str(doc_id), text=str(payload))
        if len(item) == 3:
            doc_id, second, third = item
            # Support both (doc_id, score, text_or_dict) and (doc_id, title, text)
            if isinstance(second, (int, float)):
                if isinstance(third, Mapping):
                    merged = dict(third)
                    merged.setdefault("doc_id", doc_id)
                    merged.setdefault("score", float(second))
                    return _extract_doc_fields(merged, index)
                return ContextDocument(
                    doc_id=str(doc_id),
                    text=str(third),
                    score=float(second),
                )
            return ContextDocument(
                doc_id=str(doc_id),
                title=str(second) if second is not None else None,
                text=str(third),
            )
        raise ValueError(
            f"Unsupported document tuple length ({len(item)}) at index {index}."
        )

    if isinstance(item, str):
        return ContextDocument(doc_id=default_id, text=item)

    raise TypeError(
        f"Unsupported document type at index {index}: {type(item).__name__}."
    )


def _extract_message_text(message: BaseMessage | str) -> str:
    """Extract clean plain text from a LangChain message or string."""
    if isinstance(message, str):
        return message.strip()

    # LangChain >=1.0 AIMessage exposes `.text` as a property which strips
    # non-text blocks (such as Gemini thinking parts) and joins text parts cleanly.
    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr.strip()

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


class RAGLLM:
    """LangChain + Gemini generator supporting RAG and non-RAG queries.

    Parameters
    ----------
    model:
        Gemini model identifier (defaults to ``gemini-3.5-flash-lite``, available
        on the Google AI Studio free tier).
    temperature:
        Sampling temperature. Defaults to ``0.0`` for reproducibility as
        recommended by the assignment specification.
    max_output_tokens:
        Maximum number of tokens generated in a response.
    seed:
        Optional fixed random seed for reproducibility when supported by the
        backend. Defaults to ``42``.
    max_retries:
        Automatic retry count for transient API errors (e.g., free-tier 429
        rate limits).
    timeout:
        Per-request timeout in seconds.
    language:
        Target language enforced consistently across both Condition A (no RAG)
        and Condition B (with RAG). Defaults to ``"Portuguese"``.
    api_key:
        Google AI Studio API key. If omitted, reads ``GOOGLE_API_KEY`` or
        ``GEMINI_API_KEY`` from the environment.
    top_k_context:
        Maximum number of retrieved documents included in the RAG prompt
        (defaults to ``5`` per Item 12 of the specification; pass ``None`` to
        include all provided documents).
    llm:
        Optional pre-configured LangChain ``BaseChatModel`` instance (useful
        for custom models or testing without hitting the external API).
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int | None = 1024,
        seed: int | None = 42,
        max_retries: int = 3,
        timeout: float | None = 60.0,
        language: str = "Portuguese",
        api_key: str | None = None,
        top_k_context: int | None = 5,
        llm: BaseChatModel | None = None,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")
        if top_k_context is not None and top_k_context <= 0:
            raise ValueError("top_k_context must be positive or None.")

        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.timeout = timeout
        self.language = _normalize_language(language)
        self._api_key = api_key
        self.top_k_context = top_k_context
        self._llm: BaseChatModel | None = llm

    @classmethod
    def from_config(cls, config: Any) -> "RAGLLM":
        """Instantiate :class:`RAGLLM` from a ``Config`` instance or dict."""
        getter = config.get if hasattr(config, "get") else dict.get
        return cls(
            model=getter("llm_model", DEFAULT_GEMINI_MODEL),
            temperature=getter("llm_temperature", 0.0),
            max_output_tokens=getter("llm_max_output_tokens", 1024),
            seed=getter("llm_seed", 42),
            max_retries=getter("llm_max_retries", 3),
            timeout=getter("llm_timeout", 60.0),
            language=getter("llm_language", "Portuguese"),
            api_key=getter("google_api_key", None),
            top_k_context=getter("rag_top_k", 5),
        )

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize and return the underlying LangChain chat model."""
        if self._llm is None:
            self._llm = self._build_gemini_llm()
        return self._llm

    def _build_gemini_llm(self) -> ChatGoogleGenerativeAI:
        api_key = (
            self._api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set the GOOGLE_API_KEY or "
                "GEMINI_API_KEY environment variable (free tier via Google AI "
                "Studio) or pass api_key=... to RAGLLM."
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "google_api_key": api_key,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return ChatGoogleGenerativeAI(**kwargs)

    def normalize_documents(
        self,
        documents: Sequence[Any] | None,
        top_k: int | None | object = _UNSET,
    ) -> list[ContextDocument]:
        """Normalize RAG input documents; returns ``[]`` when no RAG is used."""
        if not documents:
            return []
        if isinstance(documents, (str, bytes, Mapping)):
            raise TypeError(
                "documents must be a sequence of documents (e.g. list of dicts "
                "or LangChain Documents), or None/empty for non-RAG mode."
            )

        limit = self.top_k_context if top_k is _UNSET else top_k
        if limit is not None:
            if not isinstance(limit, int) or limit <= 0:
                raise ValueError("top_k must be a positive integer or None.")
            selected = list(documents)[:limit]
        else:
            selected = list(documents)

        return [
            _extract_doc_fields(item, idx) for idx, item in enumerate(selected)
        ]

    @staticmethod
    def format_context(documents: Sequence[ContextDocument]) -> str:
        """Format normalized documents with explicit doc_id markers for tracing."""
        blocks = [
            f"[doc_id: {doc.doc_id}]\n{doc.formatted_content()}"
            for doc in documents
        ]
        return "\n\n---\n\n".join(blocks)

    def build_prompt(
        self,
        query: str,
        documents: Sequence[Any] | None = None,
        top_k: int | None | object = _UNSET,
    ) -> tuple[ChatPromptTemplate, dict[str, Any], list[ContextDocument], Literal["A", "B"]]:
        """Build the LangChain prompt template and variables for Condition A or B."""
        cleaned_query = query.strip() if isinstance(query, str) else ""
        if not cleaned_query:
            raise ValueError("query must be a non-empty string.")

        normalized_docs = self.normalize_documents(documents, top_k=top_k)

        if not normalized_docs:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", NO_RAG_SYSTEM_PROMPT),
                    ("human", NO_RAG_HUMAN_TEMPLATE),
                ]
            )
            prompt_vars: dict[str, Any] = {
                "language": self.language,
                "query": cleaned_query,
            }
            return prompt_template, prompt_vars, normalized_docs, "A"

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", RAG_SYSTEM_PROMPT),
                ("human", RAG_HUMAN_TEMPLATE),
            ]
        )
        prompt_vars = {
            "language": self.language,
            "context": self.format_context(normalized_docs),
            "query": cleaned_query,
        }
        return prompt_template, prompt_vars, normalized_docs, "B"

    def generate_with_metadata(
        self,
        query: str,
        documents: Sequence[Any] | None = None,
        top_k: int | None | object = _UNSET,
    ) -> LLMResponse:
        """Invoke the LangChain Gemini chain and return answer plus metadata."""
        prompt_template, prompt_vars, normalized_docs, condition = self.build_prompt(
            query=query,
            documents=documents,
            top_k=top_k,
        )
        rendered_prompt = prompt_template.format(**prompt_vars)

        chain = prompt_template | self.llm
        start = time.perf_counter()
        raw_response = chain.invoke(prompt_vars)
        elapsed = time.perf_counter() - start

        answer = _extract_message_text(raw_response)
        return LLMResponse(
            query=prompt_vars["query"],
            answer=answer,
            condition=condition,
            context_doc_ids=[doc.doc_id for doc in normalized_docs],
            prompt=rendered_prompt,
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            seed=self.seed,
            language=self.language,
            latency_seconds=elapsed,
        )

    def generate(
        self,
        query: str,
        documents: Sequence[Any] | None = None,
        top_k: int | None | object = _UNSET,
    ) -> str:
        """Generate and return the answer string for the user.

        When ``documents`` is ``None`` or empty, runs in Condition A (without RAG).
        When ``documents`` is provided, runs in Condition B (with RAG).
        """
        return self.generate_with_metadata(
            query=query,
            documents=documents,
            top_k=top_k,
        ).answer

    def __call__(
        self,
        query: str,
        documents: Sequence[Any] | None = None,
        top_k: int | None | object = _UNSET,
    ) -> str:
        return self.generate(query=query, documents=documents, top_k=top_k)
