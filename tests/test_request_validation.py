from src.telegram_bot.request_validation import (
    contains_known_movie_tag,
    get_known_request_terms,
    is_valid_movie_request,
    is_vague_movie_request,
    normalize_for_matching,
    normalize_user_message,
)


def test_request_cleaning_normalizes_whitespace_and_matching_text() -> None:
    assert (
        normalize_user_message("  sci-fi\t  drama\nplease  ") == "sci-fi drama please"
    )
    assert normalize_for_matching(
        "Gruselig: Aehnlich wie \u00c4\u00d6\u00dc \u00df!"
    ) == ("gruselig aehnlich wie aeoeue ss")


def test_request_validation_uses_whole_normalized_known_terms() -> None:
    known_terms = get_known_request_terms(["Sci-Fi", "Noir", "x", "coming of age"])

    assert known_terms == {"sci fi", "noir", "coming of age"}
    assert contains_known_movie_tag("I want a sci-fi thriller", known_terms)
    assert contains_known_movie_tag("coming-of-age drama", known_terms)
    assert not contains_known_movie_tag("science documentary", known_terms)


def test_request_validation_keeps_vague_and_actionable_requests_separate() -> None:
    known_terms = {"noir"}

    assert is_vague_movie_request("movie")
    assert is_valid_movie_request("noir", known_terms)
    assert is_valid_movie_request("space mystery", known_terms)
    assert not is_valid_movie_request("?", known_terms)
    assert not is_valid_movie_request("x", known_terms)
