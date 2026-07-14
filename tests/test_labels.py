import pytest

from src.inference.labels import parse_label_vector


def test_parse_label_vector_accepts_literal_strings_and_lists() -> None:
    assert parse_label_vector("[1, 0.5, '0']", label_count=3) == [1.0, 0.5, 0.0]
    assert parse_label_vector([1, "0.25", 0], label_count=3) == [1.0, 0.25, 0.0]


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("[1, 0", "Invalid label vector literal"),
        ("(1, 0)", "Invalid label vector payload"),
        ("[1, 0, 1]", "Invalid label vector length"),
        ("[1, 'x']", "Invalid numeric label vector values"),
        ({"label": 1}, "Invalid label vector type"),
    ],
)
def test_parse_label_vector_rejects_malformed_payloads(
    value: object,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        parse_label_vector(value, label_count=2)
