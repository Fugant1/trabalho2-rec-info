"""LLM generation module using LangChain with local 4-bit Qwen3-4B (default) or Gemini.

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

import gc
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from pydantic import PrivateAttr

# Default local open-weights model quantized to 4-bit NF4 via bitsandbytes so
# it fits comfortably within 16 GB RAM and 6 GB VRAM without external API limits.
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B"
DEFAULT_LOCAL_MODEL = DEFAULT_QWEN_MODEL
DEFAULT_LLM_MODEL = DEFAULT_QWEN_MODEL
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

_UNSET = object()

_MODEL_ALIASES = {
    "qwen3:4b": DEFAULT_QWEN_MODEL,
    "qwen3-4b": DEFAULT_QWEN_MODEL,
    "qwen3 4b": DEFAULT_QWEN_MODEL,
    "qwen/qwen3-4b": DEFAULT_QWEN_MODEL,
}

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)

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
    "Answer the user's question clearly, accurately, and concisely strictly in {language}. "
    "If you are uncertain or do not have enough knowledge to answer reliably, "
    "explicitly state that."
)

NO_RAG_HUMAN_TEMPLATE = (
    "Question:\n{query}\n\n"
    "Important: Write your entire answer strictly in {language}."
)

RAG_SYSTEM_PROMPT = (
    "You are a scientific and medical information assistant operating in a "
    "Retrieval-Augmented Generation (RAG) pipeline.\n"
    "Follow these rules strictly:\n"
    "1. Write your ENTIRE answer strictly in {language}, even if the question "
    "and context documents are in another language.\n"
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
    "{query}\n\n"
    "Important: Write your ENTIRE answer strictly in {language} (do not switch "
    "to English if {language} is not English). Cite the exact [doc_id] for every "
    "claim and include a brief Sources/Fontes list at the end."
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


def _normalize_model_name(model: str) -> str:
    cleaned = model.strip()
    if not cleaned:
        raise ValueError("model must be a non-empty string.")
    return _MODEL_ALIASES.get(cleaned.lower(), cleaned)


def _normalize_language(language: str) -> str:
    cleaned = language.strip()
    if not cleaned:
        raise ValueError("language must be a non-empty string.")
    return _LANGUAGE_ALIASES.get(cleaned.lower(), cleaned)


def _strip_think_tags(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks if present."""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    return cleaned.strip()


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
        return _strip_think_tags(message)

    # LangChain >=1.0 AIMessage exposes `.text` as a property which strips
    # non-text blocks (such as thinking parts) and joins text parts cleanly.
    text_attr = getattr(message, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return _strip_think_tags(text_attr)

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return _strip_think_tags(content)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return _strip_think_tags("\n".join(part for part in parts if part))
    return _strip_think_tags(str(content))


class QuantizedLocalChatModel(BaseChatModel):
    """LangChain chat model backed by a local 4-bit quantized Hugging Face CausalLM.

    Uses ``bitsandbytes`` NF4 double quantization with ``low_cpu_mem_usage=True``
    so ``Qwen/Qwen3-4B`` runs cleanly on machines with 16 GB RAM and 6 GB VRAM.
    """

    model_name: str = DEFAULT_QWEN_MODEL
    temperature: float = 0.0
    max_output_tokens: int | None = 1024
    seed: int | None = 42
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    device_map: str = "auto"
    enable_thinking: bool = False

    _tokenizer: Any = PrivateAttr(default=None)
    _model: Any = PrivateAttr(default=None)
    _quantization_config: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        self.model_name = _normalize_model_name(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "local-transformers-4bit" if self.load_in_4bit else "local-transformers"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "seed": self.seed,
            "load_in_4bit": self.load_in_4bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
            "enable_thinking": self.enable_thinking,
        }

    @property
    def is_loaded_in_4bit(self) -> bool:
        self._ensure_loaded()
        return bool(getattr(self._model, "is_loaded_in_4bit", False))

    @property
    def quantization_config(self) -> Any:
        self._ensure_loaded()
        return self._quantization_config

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        compute_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "low_cpu_mem_usage": True,
        }
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_kwargs["quantization_config"] = bnb_config
            self._quantization_config = bnb_config
        else:
            model_kwargs["dtype"] = compute_dtype

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    def unload(self) -> None:
        """Release model weights from GPU/CPU memory."""
        self._model = None
        self._tokenizer = None
        self._quantization_config = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _format_messages(self, messages: list[BaseMessage]) -> str:
        chat_messages: list[dict[str, str]] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            chat_messages.append({"role": role, "content": _extract_message_text(msg)})

        try:
            return str(
                self._tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            )
        except TypeError:
            return str(
                self._tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        self._ensure_loaded()

        prompt_text = self._format_messages(messages)
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        target_device = getattr(self._model, "device", None)
        if target_device is not None:
            inputs = {k: v.to(target_device) for k, v in inputs.items()}

        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

        from transformers import GenerationConfig

        max_new_tokens = int(
            kwargs.get("max_new_tokens")
            or kwargs.get("max_output_tokens")
            or self.max_output_tokens
            or 1024
        )
        eos_token_id = (
            getattr(self._model.generation_config, "eos_token_id", None)
            or self._tokenizer.eos_token_id
        )
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
            "do_sample": self.temperature > 0.0,
        }
        if self.temperature > 0.0:
            gen_kwargs["temperature"] = self.temperature
        gen_config = GenerationConfig(**gen_kwargs)

        input_len = int(inputs["input_ids"].shape[-1])
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                generation_config=gen_config,
            )

        generated_ids = output_ids[0, input_len:]
        raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        cleaned_text = _strip_think_tags(raw_text)

        if stop:
            for stop_seq in stop:
                if stop_seq and stop_seq in cleaned_text:
                    cleaned_text = cleaned_text.split(stop_seq, 1)[0].strip()

        generation = ChatGeneration(message=AIMessage(content=cleaned_text))
        return ChatResult(generations=[generation])


class RAGLLM:
    """LangChain generator supporting RAG and non-RAG queries.

    Defaults to ``Qwen/Qwen3-4B`` loaded locally with 4-bit NF4 quantization
    (``bitsandbytes``) so it runs offline without external API rate limits and
    within a 16 GB RAM budget. Optionally supports Google Gemini when
    ``provider="gemini"`` or a ``gemini-*`` model name is specified.

    Parameters
    ----------
    model:
        Model identifier (defaults to ``Qwen/Qwen3-4B``; aliases such as
        ``"qwen3:4b"`` are accepted and normalized automatically).
    temperature:
        Sampling temperature. Defaults to ``0.0`` for reproducibility as
        recommended by the assignment specification.
    max_output_tokens:
        Maximum number of tokens generated in a response.
    seed:
        Optional fixed random seed for reproducibility when supported by the
        backend. Defaults to ``42``.
    max_retries:
        Automatic retry count for transient API errors (used for API backends).
    timeout:
        Per-request timeout in seconds (used for API backends).
    language:
        Target language enforced consistently across both Condition A (no RAG)
        and Condition B (with RAG). Defaults to ``"Portuguese"``.
    api_key:
        Optional Google AI Studio API key (only used if a Gemini model is selected).
    top_k_context:
        Maximum number of retrieved documents included in the RAG prompt
        (defaults to ``5`` per Item 12 of the specification; pass ``None`` to
        include all provided documents).
    provider:
        Backend selector: ``"auto"`` (default — chooses local 4-bit Transformers
        for open models like ``Qwen/Qwen3-4B`` and Gemini for ``gemini-*`` names),
        ``"local"``, or ``"gemini"``.
    load_in_4bit:
        Whether to quantize the local model to 4-bit NF4 via ``bitsandbytes``
        (defaults to ``True``).
    enable_thinking:
        Whether to enable Qwen3's internal ``<think>`` mode (defaults to
        ``False`` so generation budget goes directly to the answer).
    device_map:
        Accelerate device placement strategy for local models (default ``"auto"``).
    llm:
        Optional pre-configured LangChain ``BaseChatModel`` instance.
    """

    def __init__(
        self,
        model: str = DEFAULT_QWEN_MODEL,
        temperature: float = 0.0,
        max_output_tokens: int | None = 1024,
        seed: int | None = 42,
        max_retries: int = 3,
        timeout: float | None = 60.0,
        language: str = "Portuguese",
        api_key: str | None = None,
        top_k_context: int | None = 5,
        provider: Literal["auto", "local", "gemini"] = "auto",
        load_in_4bit: bool = True,
        enable_thinking: bool = False,
        device_map: str = "auto",
        llm: BaseChatModel | None = None,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")
        if top_k_context is not None and top_k_context <= 0:
            raise ValueError("top_k_context must be positive or None.")
        if provider not in ("auto", "local", "gemini"):
            raise ValueError("provider must be one of 'auto', 'local', or 'gemini'.")

        self.model = _normalize_model_name(model)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.seed = seed
        self.max_retries = max_retries
        self.timeout = timeout
        self.language = _normalize_language(language)
        self._api_key = api_key
        self.top_k_context = top_k_context
        self.provider = provider
        self.load_in_4bit = load_in_4bit
        self.enable_thinking = enable_thinking
        self.device_map = device_map
        self._llm: BaseChatModel | None = llm

    @classmethod
    def from_config(cls, config: Any) -> "RAGLLM":
        """Instantiate :class:`RAGLLM` from a ``Config`` instance or dict."""
        getter = config.get if hasattr(config, "get") else dict.get
        return cls(
            model=getter("llm_model", DEFAULT_QWEN_MODEL),
            temperature=getter("llm_temperature", 0.0),
            max_output_tokens=getter("llm_max_output_tokens", 1024),
            seed=getter("llm_seed", 42),
            max_retries=getter("llm_max_retries", 3),
            timeout=getter("llm_timeout", 60.0),
            language=getter("llm_language", "Portuguese"),
            api_key=getter("google_api_key", None),
            top_k_context=getter("rag_top_k", 5),
            provider=getter("llm_provider", "auto"),
            load_in_4bit=getter("llm_load_in_4bit", True),
            enable_thinking=getter("llm_enable_thinking", False),
            device_map=getter("llm_device_map", "auto"),
        )

    def _resolved_provider(self) -> Literal["local", "gemini"]:
        if self.provider in ("local", "gemini"):
            return self.provider
        if self.model.lower().startswith("gemini"):
            return "gemini"
        return "local"

    @property
    def llm(self) -> BaseChatModel:
        """Lazily initialize and return the underlying LangChain chat model."""
        if self._llm is None:
            if self._resolved_provider() == "gemini":
                self._llm = self._build_gemini_llm()
            else:
                self._llm = self._build_local_llm()
        return self._llm

    def _build_local_llm(self) -> QuantizedLocalChatModel:
        model = QuantizedLocalChatModel(
            model_name=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            seed=self.seed,
            load_in_4bit=self.load_in_4bit,
            device_map=self.device_map,
            enable_thinking=self.enable_thinking,
        )
        model._ensure_loaded()
        return model

    def _build_gemini_llm(self) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

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
        """Invoke the LangChain LLM chain and return answer plus metadata."""
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
