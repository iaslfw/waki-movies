from src.inference.title_matching import (
    build_title_candidates,
    find_referenced_movie_indices,
    get_distinctive_title_tokens,
    is_title_match_allowed,
    normalize_for_matching,
)


def test_title_normalization_handles_case_punctuation_and_german_umlauts() -> None:
    assert normalize_for_matching("\u00c4\u00d6\u00dc \u00df, Sci-Fi!") == (
        "aeoeue ss sci fi"
    )


def test_title_candidates_include_articleless_variants_and_skip_empty_titles() -> None:
    candidates = build_title_candidates([
        (10, "The Matrix"),
        (11, "Matrix"),
        (12, "A Bug's Life"),
        (13, "!!!"),
    ])

    assert ("the matrix", 10) in candidates
    assert ("matrix", 10) in candidates
    assert ("matrix", 11) in candidates
    assert ("a bug s life", 12) in candidates
    assert ("bug s life", 12) in candidates
    assert all(candidate[1] != 13 for candidate in candidates)
    assert candidates[0][0] == "a bug s life"


def test_find_referenced_movie_indices_matches_whole_titles_with_context_rules() -> (
    None
):
    candidates = build_title_candidates([
        (0, "The Matrix"),
        (1, "Her"),
        (2, "Alien"),
    ])

    assert find_referenced_movie_indices(
        "I want something like Matrix, not matrixology.",
        candidates,
        min_single_word_title_length=5,
    ) == [0]
    assert (
        find_referenced_movie_indices(
            "I liked her soundtrack.",
            candidates,
            min_single_word_title_length=5,
        )
        == []
    )
    assert find_referenced_movie_indices(
        "I saw Alien yesterday.",
        candidates,
        min_single_word_title_length=5,
    ) == [2]


def test_single_word_title_matching_requires_length_or_explicit_context() -> None:
    assert is_title_match_allowed(
        "alien",
        "alien",
        min_single_word_title_length=5,
    )
    assert is_title_match_allowed(
        "alien",
        "movies similar to alien",
        min_single_word_title_length=5,
    )
    assert not is_title_match_allowed(
        "her",
        "i liked her soundtrack",
        min_single_word_title_length=5,
    )


def test_distinctive_title_tokens_drop_articles_parts_years_and_short_tokens() -> None:
    assert get_distinctive_title_tokens(
        "The Matrix Reloaded Part 2",
        min_single_word_title_length=5,
    ) == {"matrix", "reloaded"}
