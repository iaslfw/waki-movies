"""Formatting helpers for deterministic Telegram recommendation replies."""

from collections.abc import Sequence
from typing import Any

from src.telegram_bot.language import ResponseLanguage, normalize_response_language


def format_recommendation_items(
    recommendations: list[dict[str, Any]],
    response_language: ResponseLanguage | str = "en",
    overview_texts: Sequence[str | None] | None = None,
) -> str:
    language = normalize_response_language(response_language)
    lines = []
    for index, recommendation in enumerate(recommendations):
        title = str(recommendation["title"])
        overview = get_overview_text(
            recommendation=recommendation,
            recommendation_index=index,
            response_language=language,
            overview_texts=overview_texts,
        )

        lines.append(f"{index + 1}. {title}")
        if overview:
            lines.append(overview)
        lines.append("")

    return "\n".join(lines).strip()


def get_overview_text(
    recommendation: dict[str, Any],
    recommendation_index: int,
    response_language: ResponseLanguage,
    overview_texts: Sequence[str | None] | None,
) -> str | None:
    if overview_texts is not None and recommendation_index < len(overview_texts):
        overview_text = overview_texts[recommendation_index]
        if overview_text:
            return overview_text

    if response_language == "de":
        return None

    return str(recommendation["overview"])
