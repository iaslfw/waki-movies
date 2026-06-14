"""Similarity helpers for movie recommendation ranking."""

import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


def build_user_vector(
    user_vector_dict: dict[str, float],
    movie_tags: list[str],
) -> npt.NDArray[np.float32]:
    return np.array(
        [user_vector_dict.get(tag, 0.0) for tag in movie_tags],
        dtype=np.float32,
    )


def build_positive_matrix(
    movie_matrix: npt.NDArray[np.float32],
    label_positive_threshold: float,
) -> npt.NDArray[np.float32]:
    return (movie_matrix >= label_positive_threshold).astype(np.float32)


def calculate_tag_idf_weights(
    positive_movie_matrix: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    movie_count = positive_movie_matrix.shape[0]
    positive_counts = positive_movie_matrix.sum(axis=0)
    idf_weights = np.log((movie_count + 1) / (positive_counts + 1)) + 1
    return idf_weights.astype(np.float32)


def calculate_prompt_similarities(
    user_vector: npt.NDArray[np.float32],
    movie_matrix: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    user_vector_2d: npt.NDArray[np.float32] = user_vector.reshape(1, -1)
    raw_similarities = cosine_similarity(user_vector_2d, movie_matrix)[0]
    return raw_similarities.astype(np.float32)


def calculate_weighted_tversky_similarities(
    reference_vector: npt.NDArray[np.float32],
    positive_movie_matrix: npt.NDArray[np.float32],
    tag_idf_weights: npt.NDArray[np.float32],
    *,
    label_positive_threshold: float,
    extra_tag_penalty: float,
    missing_tag_penalty: float,
) -> npt.NDArray[np.float32]:
    positive_reference = (reference_vector >= label_positive_threshold).astype(
        np.float32
    )
    weighted_movies = positive_movie_matrix * tag_idf_weights
    weighted_reference = positive_reference * tag_idf_weights

    intersection = (positive_movie_matrix * weighted_reference).sum(axis=1)
    candidate_total = weighted_movies.sum(axis=1)
    reference_total = weighted_reference.sum()
    candidate_extra = candidate_total - intersection
    reference_missing = reference_total - intersection
    denominator = (
        intersection
        + extra_tag_penalty * candidate_extra
        + missing_tag_penalty * reference_missing
    )
    similarities = np.divide(
        intersection,
        denominator,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=denominator > 0,
    )
    return similarities.astype(np.float32)


def blend_reference_and_prompt_similarities(
    reference_similarities: npt.NDArray[np.float32],
    prompt_similarities: npt.NDArray[np.float32],
    *,
    reference_weight: float,
    prompt_weight: float,
) -> npt.NDArray[np.float32]:
    similarities = reference_weight * reference_similarities
    similarities += prompt_weight * prompt_similarities
    return similarities.astype(np.float32)
