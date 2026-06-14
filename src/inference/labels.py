"""Helpers for parsing stored movie label vectors."""

import ast
from typing import Any


def parse_label_vector(value: Any, label_count: int) -> list[float]:
    """Parse a CSV label value into a fixed fallback-compatible float vector."""

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [float(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
    elif isinstance(value, list):
        return [float(item) for item in value]

    return [0.0] * label_count
