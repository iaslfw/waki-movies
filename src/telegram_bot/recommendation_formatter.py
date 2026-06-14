"""Formatting helpers for deterministic Telegram recommendation replies."""

from typing import Any


def format_recommendations(recommendations: list[dict[str, Any]]) -> str:
    lines = ["I found these movie picks for you:\n"]
    lines.append(format_recommendation_items(recommendations))
    return "\n".join(lines).strip()


def format_recommendation_items(recommendations: list[dict[str, Any]]) -> str:
    lines = []
    for i, recommendation in enumerate(recommendations, 1):
        score_percent = float(recommendation["similarity_score"]) * 100
        title = str(recommendation["title"])
        overview = str(recommendation["overview"])

        lines.append(f"{i}. {title} ({score_percent:.1f}% Match)")
        lines.append(overview)
        lines.append("")

    return "\n".join(lines).strip()
