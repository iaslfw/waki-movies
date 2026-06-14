"""Local Telegram request normalization and validation helpers."""

import re
from collections.abc import Iterable

VAGUE_SINGLE_WORD_REQUESTS = {"film", "movie", "recommendation", "something"}
KNOWN_SINGLE_WORD_REQUESTS = {
    "action",
    "abenteuer",
    "comedy",
    "dark",
    "drama",
    "familie",
    "fantasy",
    "funny",
    "gruselig",
    "horror",
    "komoedie",
    "krieg",
    "liebe",
    "romance",
    "romantic",
    "romantik",
    "romanze",
    "scary",
    "sci-fi",
    "scifi",
    "thriller",
    "war",
}
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_UMLAUT_REPLACEMENTS = (
    ("\u00e4", "ae"),
    ("\u00f6", "oe"),
    ("\u00fc", "ue"),
    ("\u00df", "ss"),
)


def normalize_user_message(message: str) -> str:
    return " ".join(message.strip().split())


def normalize_for_matching(text: str) -> str:
    lower_text = text.lower()
    for original, replacement in _UMLAUT_REPLACEMENTS:
        lower_text = lower_text.replace(original, replacement)

    alnum_text = _NON_ALNUM_PATTERN.sub(" ", lower_text)
    return " ".join(alnum_text.split())


def get_known_request_terms(movie_tags: Iterable[str]) -> set[str]:
    known_terms = {normalize_for_matching(tag) for tag in movie_tags}
    return {term for term in known_terms if len(term) >= 2}


def is_known_single_word_request(token: str, known_request_terms: set[str]) -> bool:
    return token in KNOWN_SINGLE_WORD_REQUESTS or token in known_request_terms


def contains_known_movie_tag(message: str, known_request_terms: set[str]) -> bool:
    normalized_message = normalize_for_matching(message)
    if not normalized_message:
        return False

    padded_message = f" {normalized_message} "
    return any(f" {term} " in padded_message for term in known_request_terms)


def is_valid_movie_request(message: str, known_request_terms: set[str]) -> bool:
    lower_message = message.lower()
    tokens = lower_message.split()
    meaningful_character_count = sum(char.isalnum() for char in message)
    letter_count = sum(char.isalpha() for char in message)

    if meaningful_character_count < 2 or letter_count < 2:
        return False

    if len(tokens) == 1:
        return is_known_single_word_request(
            normalize_for_matching(tokens[0]), known_request_terms
        )

    return True


def is_vague_movie_request(message: str) -> bool:
    tokens = message.lower().split()
    return len(tokens) == 1 and tokens[0] in VAGUE_SINGLE_WORD_REQUESTS
