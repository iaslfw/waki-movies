"""Helpers for parsing stored movie label vectors."""

import ast
from typing import Any


def parse_label_vector(value: Any, label_count: int) -> list[float]:
    """Parse a CSV label value into a fixed-size float vector."""

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid label vector literal: {value!r}") from exc
    elif isinstance(value, list):
        parsed = value
    else:
        raise ValueError(f"Invalid label vector type: {type(value).__name__}")

    if not isinstance(parsed, list):
        raise ValueError(f"Invalid label vector payload: {parsed!r}")

    if len(parsed) != label_count:
        raise ValueError(
            f"Invalid label vector length: expected {label_count}, got {len(parsed)}"
        )

    try:
        return [float(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric label vector values: {parsed!r}") from exc
