"""Spelling correction (error tolerance) module using k-grams, Jaccard, and edit distance.

Implements Requirement 4 of Trabalho Pratico 2:
- Builds a dictionary from collection terms BEFORE stemming/lemmatization
  (including stopwords so valid query stopwords are never falsely corrected).
- Generates candidates for unknown query terms via a character k-gram index.
- Filters candidates using the Jaccard coefficient over k-gram sets.
- Selects the closest correction using edit distance (Damerau-Levenshtein or
  standard Levenshtein), breaking ties by Jaccard similarity and corpus
  frequency.
- Provides utilities to construct, validate, and dev-tune typo experiments
  covering insertion, deletion, substitution, and transposition errors.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ERROR_TYPES: tuple[str, ...] = ("insercao", "remocao", "substituicao", "troca")

_KEYBOARD_NEIGHBORS: dict[str, str] = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "serfcx",
    "e": "wsdr",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "i": "ujko",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "m": "njk",
    "n": "bhjm",
    "o": "iklp",
    "p": "ol",
    "q": "wa",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "u": "yhji",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "tghu",
    "z": "asx",
}


def extract_kgrams(term: str, k: int = 2, pad: bool = True) -> set[str]:
    """Extract character k-grams from a term, optionally padded with ``$``."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    padded = f"${term}$" if pad else term
    if len(padded) < k:
        return {padded} if padded else set()
    return {padded[i : i + k] for i in range(len(padded) - k + 1)}


def jaccard_coefficient(grams_a: set[str], grams_b: set[str]) -> float:
    """Compute Jaccard similarity ``|A ∩ B| / |A ∪ B|`` between two k-gram sets."""
    if not grams_a and not grams_b:
        return 0.0
    intersection = len(grams_a & grams_b)
    union = len(grams_a) + len(grams_b) - intersection
    return (intersection / union) if union > 0 else 0.0


def levenshtein_distance(a: str, b: str) -> int:
    """Compute standard Levenshtein edit distance (insert, delete, substitute = 1)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, ch_b in enumerate(b, start=1):
            cost = 0 if ch_a == ch_b else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev = curr
    return prev[-1]


def damerau_levenshtein_distance(a: str, b: str) -> int:
    """Compute Optimal String Alignment (Damerau-Levenshtein) edit distance.

    Supports insertion, deletion, substitution, and adjacent transposition at cost 1.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j

    for i in range(1, la + 1):
        ch_a = a[i - 1]
        for j in range(1, lb + 1):
            ch_b = b[j - 1]
            cost = 0 if ch_a == ch_b else 1
            best = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost, # substitution
            )
            if (
                i > 1
                and j > 1
                and ch_a == b[j - 2]
                and a[i - 2] == ch_b
            ):
                best = min(best, dp[i - 2][j - 2] + 1)  # transposition
            dp[i][j] = best

    return dp[la][lb]


def _extract_raw_doc_text(raw_doc: Any) -> str:
    if isinstance(raw_doc, Mapping):
        title = str(raw_doc.get("title") or "").strip()
        text = str(raw_doc.get("text") or "").strip()
        return f"{title} {text}".strip() if (title and text) else (title or text)
    return str(raw_doc)


class SpellCorrector:
    """From-scratch spell corrector using k-grams, Jaccard filtering, and edit distance."""

    def __init__(
        self,
        k: int = 2,
        jaccard_threshold: float = 0.25,
        max_edit_distance: int = 2,
        max_candidates: int | None = 100,
        distance_metric: str = "damerau_levenshtein",
        pad_kgrams: bool = True,
        min_term_length: int = 2,
        token_pattern: str = r"[a-z0-9]+",
    ) -> None:
        if k < 1:
            raise ValueError("k must be at least 1.")
        if not 0.0 <= jaccard_threshold <= 1.0:
            raise ValueError("jaccard_threshold must be between 0.0 and 1.0.")
        if max_edit_distance < 1:
            raise ValueError("max_edit_distance must be at least 1.")
        if distance_metric not in ("levenshtein", "damerau_levenshtein"):
            raise ValueError(
                "distance_metric must be 'levenshtein' or 'damerau_levenshtein'."
            )

        self.k = k
        self.jaccard_threshold = jaccard_threshold
        self.max_edit_distance = max_edit_distance
        self.max_candidates = max_candidates
        self.distance_metric = distance_metric
        self.pad_kgrams = pad_kgrams
        self.min_term_length = min_term_length
        self.token_pattern = token_pattern
        self._regex = re.compile(token_pattern, re.IGNORECASE)

        self.vocabulary: set[str] = set()
        self.term_frequencies: dict[str, int] = {}
        self.kgram_index: dict[str, set[str]] = {}
        self._term_kgrams: dict[str, set[str]] = {}

    def _edit_distance(self, a: str, b: str) -> int:
        if self.distance_metric == "levenshtein":
            return levenshtein_distance(a, b)
        return damerau_levenshtein_distance(a, b)

    def fit_from_corpus(
        self,
        documents: Mapping[str, Any] | Iterable[str],
    ) -> "SpellCorrector":
        """Build dictionary from raw corpus texts BEFORE stemming/lemmatization.

        All alphanumeric tokens (lowercased, including stopwords) are retained
        so that stopwords in queries are recognized as known words and never
        miscorrected.
        """
        counts: Counter[str] = Counter()
        if isinstance(documents, Mapping):
            texts = (_extract_raw_doc_text(doc) for doc in documents.values())
        else:
            texts = (_extract_raw_doc_text(doc) for doc in documents)

        for text in texts:
            doc_tokens = set(self._regex.findall(text.lower()))
            counts.update(doc_tokens)

        return self.fit_from_vocabulary(counts, include_english_stopwords=True)

    def fit_from_vocabulary(
        self,
        vocab: Mapping[str, int] | Iterable[str],
        include_english_stopwords: bool = True,
    ) -> "SpellCorrector":
        """Build k-gram index directly from a vocabulary (or term -> freq mapping)."""
        self.vocabulary.clear()
        self.term_frequencies.clear()
        self.kgram_index.clear()
        self._term_kgrams.clear()

        if isinstance(vocab, Mapping):
            for term, freq in vocab.items():
                cleaned = str(term).lower().strip()
                if cleaned:
                    self.term_frequencies[cleaned] = max(int(freq), 1)
        else:
            for term in vocab:
                cleaned = str(term).lower().strip()
                if cleaned:
                    self.term_frequencies[cleaned] = self.term_frequencies.get(cleaned, 0) + 1

        if include_english_stopwords:
            for sw in ENGLISH_STOP_WORDS:
                self.term_frequencies.setdefault(sw, 1)

        self.vocabulary = set(self.term_frequencies.keys())

        for term in self.vocabulary:
            grams = extract_kgrams(term, k=self.k, pad=self.pad_kgrams)
            self._term_kgrams[term] = grams
            for gram in grams:
                self.kgram_index.setdefault(gram, set()).add(term)

        return self

    def fit(
        self,
        source: Mapping[str, Any] | Iterable[str],
        *,
        is_corpus: bool | None = None,
    ) -> "SpellCorrector":
        """Fit dictionary from either a corpus mapping/iterable or a vocabulary."""
        if is_corpus is True:
            return self.fit_from_corpus(source)
        if is_corpus is False:
            return self.fit_from_vocabulary(source)

        if isinstance(source, Mapping):
            sample_val = next(iter(source.values()), None)
            if isinstance(sample_val, (int, float)):
                return self.fit_from_vocabulary(source)
            return self.fit_from_corpus(source)
        return self.fit_from_vocabulary(source)

    def is_known(self, term: str) -> bool:
        return term.lower() in self.vocabulary

    def generate_candidates(self, term: str) -> list[tuple[str, float]]:
        """Generate candidate terms via k-grams and filter by Jaccard threshold."""
        cleaned = term.lower()
        query_grams = extract_kgrams(cleaned, k=self.k, pad=self.pad_kgrams)
        if not query_grams:
            return []

        raw_candidates: set[str] = set()
        for gram in query_grams:
            posting = self.kgram_index.get(gram)
            if posting:
                raw_candidates.update(posting)

        filtered: list[tuple[str, float]] = []
        max_len_diff = self.max_edit_distance
        q_len = len(cleaned)
        for cand in raw_candidates:
            # Exact edit distance lower bound: |len(a) - len(b)| <= edit_distance
            if abs(len(cand) - q_len) > max_len_diff:
                continue
            jacc = jaccard_coefficient(query_grams, self._term_kgrams[cand])
            if jacc >= self.jaccard_threshold:
                filtered.append((cand, jacc))

        filtered.sort(
            key=lambda item: (-item[1], -self.term_frequencies.get(item[0], 0), item[0])
        )
        if self.max_candidates is not None:
            filtered = filtered[: self.max_candidates]
        return filtered

    def correct_term(self, term: str) -> dict[str, Any]:
        """Correct a single token if unknown, returning rich diagnostic metadata."""
        cleaned = term.lower()
        if (
            not cleaned
            or len(cleaned) < self.min_term_length
            or cleaned.isdigit()
            or self.is_known(cleaned)
        ):
            return {
                "original": term,
                "corrected": term,
                "was_unknown": False,
                "was_corrected": False,
                "edit_distance": 0,
                "jaccard": 1.0 if self.is_known(cleaned) else None,
                "candidates_checked": 0,
                "top_candidates": [],
            }

        candidates = self.generate_candidates(cleaned)
        scored: list[tuple[int, float, int, str]] = []
        for cand, jacc in candidates:
            dist = self._edit_distance(cleaned, cand)
            if dist <= self.max_edit_distance:
                scored.append(
                    (dist, -jacc, -self.term_frequencies.get(cand, 0), cand)
                )

        if not scored:
            return {
                "original": term,
                "corrected": term,
                "was_unknown": True,
                "was_corrected": False,
                "edit_distance": None,
                "jaccard": None,
                "candidates_checked": len(candidates),
                "top_candidates": [],
            }

        scored.sort()
        best_dist, neg_jacc, _, best_term = scored[0]
        top_candidates = [
            {
                "term": c_term,
                "edit_distance": c_dist,
                "jaccard": round(-c_neg_jacc, 6),
                "freq": -c_neg_freq,
            }
            for c_dist, c_neg_jacc, c_neg_freq, c_term in scored[:5]
        ]
        return {
            "original": term,
            "corrected": best_term,
            "was_unknown": True,
            "was_corrected": best_term != cleaned,
            "edit_distance": best_dist,
            "jaccard": -neg_jacc,
            "candidates_checked": len(candidates),
            "top_candidates": top_candidates,
        }

    def correct_query_with_details(self, query: str) -> dict[str, Any]:
        """Correct unknown tokens in a query while preserving punctuation and spacing."""
        corrections: list[dict[str, Any]] = []

        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            info = self.correct_term(token)
            if info["was_unknown"]:
                corrections.append(info)
            return info["corrected"]

        corrected_query = self._regex.sub(_replace, query)
        return {
            "original_query": query,
            "corrected_query": corrected_query,
            "corrections": corrections,
            "num_unknown_tokens": len(corrections),
            "num_corrected_tokens": sum(1 for c in corrections if c["was_corrected"]),
        }

    def correct(self, text: str) -> str:
        """Return the spelling-corrected query string."""
        return self.correct_query_with_details(text)["corrected_query"]

    def validate_typo_pair(
        self,
        original_query: str,
        typo_query: str,
    ) -> dict[str, Any]:
        """Validate Requirement 4-(ii): typo must produce at least one OOV term."""
        orig_tokens = [t.lower() for t in self._regex.findall(original_query)]
        typo_tokens = [t.lower() for t in self._regex.findall(typo_query)]
        orig_set = set(orig_tokens)
        oov_tokens = [
            t for t in typo_tokens if not self.is_known(t) and t not in orig_set
        ]
        changed_in_vocab = [
            t for t in typo_tokens if self.is_known(t) and t not in orig_set
        ]
        is_valid = (original_query.strip() != typo_query.strip()) and (len(oov_tokens) > 0)
        return {
            "is_valid": is_valid,
            "oov_tokens": oov_tokens,
            "accidental_valid_words": changed_in_vocab,
        }

    def get_metadata(self) -> dict[str, Any]:
        return {
            "k_gram": self.k,
            "jaccard_threshold": self.jaccard_threshold,
            "max_edit_distance": self.max_edit_distance,
            "max_candidates": self.max_candidates,
            "distance_metric": self.distance_metric,
            "pad_kgrams": self.pad_kgrams,
            "vocabulary_size": len(self.vocabulary),
            "num_kgrams": len(self.kgram_index),
        }


class TypoHandler:
    """High-level typo handler compatible with custom models or :class:`SpellCorrector`."""

    def __init__(self, typo_correction_model: Any = None) -> None:
        self.typo_correction_model = typo_correction_model

    def correct_typos(self, text: str) -> str:
        if self.typo_correction_model is None:
            raise ValueError("typo_correction_model is not initialized.")
        return self.typo_correction_model.correct(text)

    def correct_with_details(self, text: str) -> dict[str, Any]:
        if self.typo_correction_model is None:
            raise ValueError("typo_correction_model is not initialized.")
        if hasattr(self.typo_correction_model, "correct_query_with_details"):
            return self.typo_correction_model.correct_query_with_details(text)
        corrected = self.typo_correction_model.correct(text)
        return {
            "original_query": text,
            "corrected_query": corrected,
            "corrections": [],
        }


def introduce_typo(
    word: str,
    error_type: str,
    vocabulary: set[str],
    rng: random.Random,
) -> str | None:
    """Introduce a single guaranteed-OOV typo of ``error_type`` into ``word``.

    Supported ``error_type`` values:
    - ``"insercao"``: insert an adjacent/nearby letter
    - ``"remocao"``: delete an interior character
    - ``"substituicao"``: replace a character with a keyboard neighbor
    - ``"troca"``: swap two adjacent distinct characters
    """
    cleaned = word.lower()
    if len(cleaned) < 4 or not cleaned.isalpha():
        return None
    if error_type not in ERROR_TYPES:
        raise ValueError(f"Unsupported error_type '{error_type}'. Expected one of {ERROR_TYPES}.")

    candidates: list[str] = []

    if error_type == "insercao":
        positions = list(range(1, len(cleaned)))
        rng.shuffle(positions)
        for pos in positions:
            ch = cleaned[pos - 1]
            neighbors = _KEYBOARD_NEIGHBORS.get(ch, "aeiourst")
            for ins in neighbors:
                cand = cleaned[:pos] + ins + cleaned[pos:]
                candidates.append(cand)
    elif error_type == "remocao":
        positions = list(range(1, len(cleaned) - 1))
        rng.shuffle(positions)
        for pos in positions:
            cand = cleaned[:pos] + cleaned[pos + 1 :]
            candidates.append(cand)
    elif error_type == "substituicao":
        positions = list(range(1, len(cleaned) - 1))
        rng.shuffle(positions)
        for pos in positions:
            ch = cleaned[pos]
            neighbors = _KEYBOARD_NEIGHBORS.get(ch, "aeiourst")
            for rep in neighbors:
                if rep != ch:
                    cand = cleaned[:pos] + rep + cleaned[pos + 1 :]
                    candidates.append(cand)
    elif error_type == "troca":
        positions = [i for i in range(len(cleaned) - 1) if cleaned[i] != cleaned[i + 1]]
        rng.shuffle(positions)
        for pos in positions:
            cand = (
                cleaned[:pos]
                + cleaned[pos + 1]
                + cleaned[pos]
                + cleaned[pos + 2 :]
            )
            candidates.append(cand)

    for cand in candidates:
        if cand != cleaned and cand not in vocabulary:
            return cand
    return None


def build_typo_benchmark(
    queries: Mapping[str, str],
    eligible_query_ids: Sequence[str],
    vocabulary: set[str],
    num_queries: int = 10,
    seed: int = 42,
    error_types: Sequence[str] = ERROR_TYPES,
    token_pattern: str = r"[a-z0-9]+",
) -> list[dict[str, str]]:
    """Deterministically build ``num_queries`` valid OOV typo cases from eligible queries.

    Selects queries with meaningful content words (non-stopwords, alphabetic,
    length >= 5, present in vocabulary) and balances the 4 error types
    (insercao, remocao, substituicao, troca).
    """
    rng = random.Random(seed)
    token_re = re.compile(token_pattern, re.IGNORECASE)

    # Filter eligible queries that have at least one suitable content token
    candidates_with_tokens: list[tuple[str, str, list[re.Match[str]]]] = []
    for qid in sorted(eligible_query_ids):
        qtext = str(queries[qid])
        matches = [
            m
            for m in token_re.finditer(qtext)
            if (
                len(m.group(0)) >= 5
                and m.group(0).isalpha()
                and m.group(0).lower() not in ENGLISH_STOP_WORDS
                and m.group(0).lower() in vocabulary
            )
        ]
        if matches:
            candidates_with_tokens.append((qid, qtext, matches))

    rng.shuffle(candidates_with_tokens)

    benchmark: list[dict[str, str]] = []
    type_cycle = list(error_types)

    for qid, qtext, matches in candidates_with_tokens:
        if len(benchmark) >= num_queries:
            break
        target_type = type_cycle[len(benchmark) % len(type_cycle)]

        # Prefer longer (more specific) domain words first
        ordered_matches = sorted(matches, key=lambda m: -len(m.group(0)))
        corrupted_query: str | None = None
        corrupted_word: str | None = None
        original_word: str | None = None

        for match in ordered_matches:
            orig_tok = match.group(0)
            typo_tok = introduce_typo(orig_tok, target_type, vocabulary, rng)
            if typo_tok is not None:
                corrupted_query = (
                    qtext[: match.start()] + typo_tok + qtext[match.end() :]
                )
                original_word = orig_tok
                corrupted_word = typo_tok
                break

        if corrupted_query is not None:
            benchmark.append(
                {
                    "query_id": str(qid),
                    "query_original": qtext,
                    "query_erro": corrupted_query,
                    "tipo_erro": target_type,
                    "termo_original": str(original_word),
                    "termo_erro": str(corrupted_word),
                }
            )

    if len(benchmark) < num_queries:
        raise ValueError(
            f"Could only construct {len(benchmark)} valid typo queries out of "
            f"requested {num_queries}."
        )
    return benchmark


def tune_spell_corrector_on_dev(
    corpus_documents: Mapping[str, Any],
    dev_queries: Mapping[str, str],
    dev_qrels: Mapping[str, Mapping[str, int | float]],
    lexical_retriever: Any,
    evaluator: Any,
    k_values: Sequence[int] = (2, 3),
    jaccard_thresholds: Sequence[float] = (0.2, 0.25, 0.35),
    max_edit_distances: Sequence[int] = (2,),
    distance_metric: str = "damerau_levenshtein",
    num_dev_queries: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Tune SpellCorrector hyperparameters on the ``dev`` split (Requirement 4).

    Builds a synthetic OOV typo benchmark on eligible dev queries (those with
    lexical ``NDCG@10 > 0``) and selects the ``(k, jaccard_threshold,
    max_edit_distance)`` configuration with the highest mean corrected
    ``NDCG@10`` (breaking ties by exact term recovery accuracy).
    """
    base_corrector = SpellCorrector(k=2).fit_from_corpus(corpus_documents)

    dev_qids = sorted(dev_qrels.keys())
    subset_queries = {qid: dev_queries[qid] for qid in dev_qids if qid in dev_queries}
    orig_rankings = lexical_retriever.retrieve(subset_queries, top_k=100)
    orig_eval = evaluator.evaluate(dev_qrels, orig_rankings, k=10)
    eligible = [
        qid for qid, m in orig_eval["per_query"].items() if m["NDCG10"] > 0.0
    ]
    sample_n = min(num_dev_queries, len(eligible))
    if sample_n == 0:
        raise ValueError("No eligible dev queries with NDCG@10 > 0 for typo tuning.")

    dev_cases = build_typo_benchmark(
        subset_queries,
        eligible,
        base_corrector.vocabulary,
        num_queries=sample_n,
        seed=seed,
    )

    err_ndcg_by_qid: dict[str, float] = {}
    orig_ndcg_by_qid: dict[str, float] = {}
    for case in dev_cases:
        qid = case["query_id"]
        orig_ndcg_by_qid[qid] = orig_eval["per_query"][qid]["NDCG10"]
        err_rank = lexical_retriever.search(case["query_erro"], top_k=100)
        err_ndcg_by_qid[qid] = evaluator.evaluate_query(
            dev_qrels[qid], err_rank, k=10
        )["NDCG10"]

    grid_results: list[dict[str, Any]] = []
    best_cfg: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None

    for k in k_values:
        # Build k-gram index once per k value, reuse across thresholds/distances
        template = SpellCorrector(
            k=k,
            jaccard_threshold=0.0,
            max_edit_distance=max(max_edit_distances),
            distance_metric=distance_metric,
        ).fit_from_vocabulary(
            base_corrector.term_frequencies,
            include_english_stopwords=False,
        )

        for jacc in jaccard_thresholds:
            for max_ed in max_edit_distances:
                template.jaccard_threshold = jacc
                template.max_edit_distance = max_ed

                corr_ndcgs: list[float] = []
                recoveries: list[float] = []
                exact_word_hits = 0

                for case in dev_cases:
                    qid = case["query_id"]
                    corr_query = template.correct(case["query_erro"])
                    if corr_query.lower() == case["query_original"].lower():
                        exact_word_hits += 1
                    corr_rank = lexical_retriever.search(corr_query, top_k=100)
                    corr_ndcg = evaluator.evaluate_query(
                        dev_qrels[qid], corr_rank, k=10
                    )["NDCG10"]
                    corr_ndcgs.append(corr_ndcg)

                    rec_info = evaluator.compute_typo_recovery(
                        orig_ndcg_by_qid[qid],
                        err_ndcg_by_qid[qid],
                        corr_ndcg,
                    )
                    if rec_info["recovery_rate"] is not None:
                        recoveries.append(rec_info["recovery_rate"])

                mean_corr_ndcg = sum(corr_ndcgs) / len(corr_ndcgs)
                mean_recovery = (
                    sum(recoveries) / len(recoveries) if recoveries else 0.0
                )
                exact_acc = exact_word_hits / len(dev_cases)

                entry = {
                    "k": k,
                    "jaccard_threshold": jacc,
                    "max_edit_distance": max_ed,
                    "distance_metric": distance_metric,
                    "mean_ndcg10_original": sum(orig_ndcg_by_qid.values()) / len(dev_cases),
                    "mean_ndcg10_erro": sum(err_ndcg_by_qid.values()) / len(dev_cases),
                    "mean_ndcg10_corrigida": mean_corr_ndcg,
                    "mean_recovery_rate": mean_recovery,
                    "exact_query_accuracy": exact_acc,
                    "num_dev_cases": len(dev_cases),
                }
                grid_results.append(entry)

                rank_key = (round(mean_corr_ndcg, 8), round(exact_acc, 8), round(mean_recovery, 8))
                if best_key is None or rank_key > best_key:
                    best_key = rank_key
                    best_cfg = entry

    assert best_cfg is not None
    return {
        "best_params": {
            "k": best_cfg["k"],
            "jaccard_threshold": best_cfg["jaccard_threshold"],
            "max_edit_distance": best_cfg["max_edit_distance"],
            "distance_metric": best_cfg["distance_metric"],
        },
        "best_result": best_cfg,
        "grid_results": grid_results,
    }
