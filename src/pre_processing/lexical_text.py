"""Shared tokenization for lexical indexing and retrieval."""

from __future__ import annotations

from collections.abc import Callable, Collection
from importlib.metadata import version
from re import Pattern

STEMMING_MODES = ("none", "snowball-english")


def stemming_metadata(mode: str) -> dict[str, str]:
    """Identify the stemming configuration used to build an index."""
    if mode == "none":
        return {"stemming": mode}
    if mode != "snowball-english":
        raise ValueError(f"Unknown lexical stemming mode: {mode!r}")
    return {
        "stemming": mode,
        "stemmer_language": "english",
        "stemmer_version": version("snowballstemmer"),
    }


def create_stemmer(mode: str) -> Callable[[str], str] | None:
    """Return a stemmer for a lexical preprocessing mode."""
    if mode == "none":
        return None
    if mode != "snowball-english":
        raise ValueError(f"Unknown lexical stemming mode: {mode!r}")

    from snowballstemmer import stemmer

    return stemmer("english").stemWord


def tokenize_lexical(
    text: str,
    *,
    lowercase: bool,
    pattern: Pattern[str],
    min_token_length: int,
    stopwords: Collection[str] | None,
    stem_word: Callable[[str], str] | None,
) -> list[str]:
    """Apply lexical filters and optionally stem lowercase ASCII words."""
    normalized = text.lower() if lowercase else text
    tokens = pattern.findall(normalized)
    if min_token_length > 1:
        tokens = [token for token in tokens if len(token) >= min_token_length]
    if stopwords is not None:
        tokens = [token for token in tokens if token.lower() not in stopwords]
    if stem_word is not None:
        tokens = [
            stem_word(token) if token.isascii() and token.isalpha() and token.islower() else token
            for token in tokens
        ]
    return tokens
