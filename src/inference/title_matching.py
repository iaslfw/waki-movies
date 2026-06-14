"""Movie-title normalization and matching helpers."""

import re
from collections.abc import Iterable
from typing import Any

TitleCandidate = tuple[str, int]

_TITLE_ARTICLES = ("the ", "a ", "an ")
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_UMLAUT_REPLACEMENTS = (
    ("\u00e4", "ae"),
    ("\u00f6", "oe"),
    ("\u00fc", "ue"),
    ("\u00df", "ss"),
)
_IGNORED_TITLE_TOKENS = {
    "a",
    "an",
    "and",
    "der",
    "die",
    "das",
    "le",
    "la",
    "of",
    "part",
    "the",
}


def normalize_for_matching(text: str) -> str:
    lower_text = text.casefold()
    for original, replacement in _UMLAUT_REPLACEMENTS:
        lower_text = lower_text.replace(original, replacement)

    alnum_text = _NON_ALNUM_PATTERN.sub(" ", lower_text)
    return " ".join(alnum_text.split())


def build_title_candidates(titles: Iterable[tuple[Any, Any]]) -> list[TitleCandidate]:
    candidates: list[TitleCandidate] = []
    seen: set[TitleCandidate] = set()
    for idx, title in titles:
        normalized_title = normalize_for_matching(str(title))
        if not normalized_title:
            continue

        variants = {normalized_title}
        for article in _TITLE_ARTICLES:
            if normalized_title.startswith(article):
                variants.add(normalized_title.removeprefix(article))

        for variant in variants:
            title_key = (variant, int(idx))
            if title_key in seen:
                continue

            seen.add(title_key)
            candidates.append(title_key)

    return sorted(
        candidates,
        key=lambda item: (len(item[0].split()), len(item[0])),
        reverse=True,
    )


def find_referenced_movie_indices(
    query: str,
    title_candidates: list[TitleCandidate],
    *,
    min_single_word_title_length: int,
) -> list[int]:
    normalized_query = normalize_for_matching(query)
    if not normalized_query:
        return []

    padded_query = f" {normalized_query} "
    matches: list[int] = []
    consumed_titles: set[str] = set()
    for normalized_title, idx in title_candidates:
        if normalized_title in consumed_titles:
            continue
        if not is_title_match_allowed(
            normalized_title,
            normalized_query,
            min_single_word_title_length=min_single_word_title_length,
        ):
            continue
        if f" {normalized_title} " not in padded_query:
            continue

        matches.append(idx)
        consumed_titles.add(normalized_title)

    return matches


def is_title_match_allowed(
    normalized_title: str,
    normalized_query: str,
    *,
    min_single_word_title_length: int,
) -> bool:
    title_tokens = normalized_title.split()
    if len(title_tokens) > 1:
        return True

    title = title_tokens[0]
    if len(title) < min_single_word_title_length:
        return False
    if normalized_query == title:
        return True

    title_context_patterns = (
        f"like {title}",
        f"similar to {title}",
        f"watched {title}",
        f"saw {title}",
        f"wie {title}",
        f"aehnlich {title}",
        f"aehnlich wie {title}",
        f"gesehen {title}",
        f"geschaut {title}",
    )
    return any(pattern in normalized_query for pattern in title_context_patterns)


def get_distinctive_title_tokens(
    title: str,
    *,
    min_single_word_title_length: int,
) -> set[str]:
    return {
        token
        for token in normalize_for_matching(title).split()
        if len(token) >= min_single_word_title_length
        and token not in _IGNORED_TITLE_TOKENS
        and not token.isdigit()
    }
