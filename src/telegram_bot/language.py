"""Language value helpers for structured Mistral responses."""

from typing import Literal

ResponseLanguage = Literal["en", "de"]


def normalize_response_language(value: object) -> ResponseLanguage:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "de":
            return "de"
        if normalized == "en":
            return "en"

    return "en"
