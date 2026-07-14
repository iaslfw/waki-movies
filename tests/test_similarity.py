import numpy as np

from src.inference.similarity import (
    blend_reference_and_prompt_similarities,
    build_positive_matrix,
    build_user_vector,
    calculate_prompt_similarities,
    calculate_tag_idf_weights,
    calculate_weighted_tversky_similarities,
)


def test_build_user_vector_preserves_tag_order_and_fills_missing_values() -> None:
    user_vector = build_user_vector(
        {"dark": 0.8, "funny": 0.2},
        ["funny", "scary", "dark"],
    )

    np.testing.assert_array_equal(
        user_vector,
        np.asarray([0.2, 0.0, 0.8], dtype=np.float32),
    )
    assert user_vector.dtype == np.float32


def test_positive_matrix_threshold_is_inclusive() -> None:
    movie_matrix = np.asarray(
        [
            [0.69, 0.70, 0.71],
            [0.00, 1.00, 0.50],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(
        build_positive_matrix(movie_matrix, label_positive_threshold=0.70),
        np.asarray([[0, 1, 1], [0, 1, 0]], dtype=np.float32),
    )


def test_idf_weights_follow_smoothed_inverse_frequency_formula() -> None:
    positive_movie_matrix = np.asarray(
        [
            [1, 1, 0],
            [1, 0, 0],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )

    weights = calculate_tag_idf_weights(positive_movie_matrix)

    expected = np.log((3 + 1) / (np.asarray([3, 1, 1]) + 1)) + 1
    np.testing.assert_allclose(weights, expected.astype(np.float32))
    assert weights[0] < weights[1]


def test_prompt_similarities_use_cosine_similarity() -> None:
    similarities = calculate_prompt_similarities(
        np.asarray([1, 0], dtype=np.float32),
        np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
    )

    np.testing.assert_allclose(
        similarities,
        np.asarray([1.0, 0.0, 2**-0.5], dtype=np.float32),
    )


def test_weighted_tversky_penalizes_extra_and_missing_tags() -> None:
    similarities = calculate_weighted_tversky_similarities(
        reference_vector=np.asarray([0.9, 0.1, 0.8], dtype=np.float32),
        positive_movie_matrix=np.asarray(
            [
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.float32,
        ),
        tag_idf_weights=np.asarray([1, 2, 3], dtype=np.float32),
        label_positive_threshold=0.7,
        extra_tag_penalty=1.5,
        missing_tag_penalty=0.5,
    )

    np.testing.assert_allclose(
        similarities,
        np.asarray([1.0, 1.0 / 5.5, 0.0, 0.0], dtype=np.float32),
    )


def test_weighted_tversky_returns_zero_for_empty_denominator() -> None:
    similarities = calculate_weighted_tversky_similarities(
        reference_vector=np.asarray([0.0, 0.0], dtype=np.float32),
        positive_movie_matrix=np.asarray([[0, 0]], dtype=np.float32),
        tag_idf_weights=np.asarray([1, 1], dtype=np.float32),
        label_positive_threshold=0.7,
        extra_tag_penalty=1.5,
        missing_tag_penalty=0.5,
    )

    np.testing.assert_array_equal(similarities, np.asarray([0.0], dtype=np.float32))


def test_blend_reference_and_prompt_similarities_uses_configured_weights() -> None:
    blended = blend_reference_and_prompt_similarities(
        reference_similarities=np.asarray([1.0, 0.2], dtype=np.float32),
        prompt_similarities=np.asarray([0.0, 0.8], dtype=np.float32),
        reference_weight=0.7,
        prompt_weight=0.3,
    )

    np.testing.assert_allclose(blended, np.asarray([0.7, 0.38], dtype=np.float32))
