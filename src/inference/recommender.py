"""
Recommender module for finding similar movies based on movie-tag vectors.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.inference.labels import parse_label_vector
from src.inference.similarity import (
    blend_reference_and_prompt_similarities,
    build_positive_matrix,
    build_user_vector,
    calculate_prompt_similarities,
    calculate_tag_idf_weights,
    calculate_weighted_tversky_similarities,
)
from src.inference.title_matching import (
    build_title_candidates,
    find_referenced_movie_indices,
    get_distinctive_title_tokens,
    is_title_match_allowed,
    normalize_for_matching as normalize_title_for_matching,
)
from src.settings import Settings


class MovieRecommender:
    df: pd.DataFrame
    movie_matrix: npt.NDArray[np.float32]
    positive_movie_matrix: npt.NDArray[np.float32]
    tag_idf_weights: npt.NDArray[np.float32]
    movie_tags: list[str]
    title_candidates: list[tuple[str, int]]

    def __init__(self, data_path: str | Path) -> None:
        """
        Loads movie data and prepares the movie-tag vectors for recommendations

        Args:
            data_path: Path to the CSV file containing movie data.
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"No file found under: {path}.")

        self.movie_tags = Settings.create_movie_tag_list()

        self.df = pd.read_csv(path)

        print("Create movie-tag vectors...")

        label_count = len(self.movie_tags)
        self.df["labels"] = self.df["labels"].apply(
            lambda value: parse_label_vector(value, label_count)
        )

        self.movie_matrix = np.array(self.df["labels"].tolist(), dtype=np.float32)
        self.positive_movie_matrix = build_positive_matrix(
            self.movie_matrix,
            Settings.LABEL_POSITIVE_THRESHOLD,
        )
        self.tag_idf_weights = self._calculate_tag_idf_weights()
        self.title_candidates = build_title_candidates(self.df["title"].items())

        print(f"Recommender ready! {len(self.df)} movies loaded.")

    def get_recommendations(
        self,
        user_vector_dict: dict[str, float],
        top_k: int = 3,
        query: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compares user vector with all movies, returns the top k most similiar movies.

        Args:
            user_vector_dict: A dictionary mapping movie tags to their corresponding values for the user.
            top_k: The number of top recommendations to return.
            query: Optional raw user query used to detect referenced movie titles.
        """
        user_vector = build_user_vector(user_vector_dict, self.movie_tags)

        reference_indices = self.find_referenced_movie_indices(query or "")
        excluded_indices = set(reference_indices)
        if reference_indices:
            similarities = self._calculate_reference_movie_similarities(
                user_vector=user_vector,
                reference_indices=reference_indices,
            )
        else:
            similarities = self._calculate_prompt_similarities(user_vector)

        sorted_indices: npt.NDArray[np.intp] = np.argsort(similarities)[::-1]

        recommendations: list[dict[str, Any]] = []
        for idx_val in sorted_indices:
            idx = int(idx_val)
            if idx in excluded_indices:
                continue

            title = str(self.df.at[idx, "title"])
            overview = str(self.df.at[idx, "text"])

            recommendations.append({
                "title": title,
                "overview": overview,
                "similarity_score": float(similarities[idx]),
            })
            if len(recommendations) >= top_k:
                break

        return recommendations

    def _calculate_tag_idf_weights(self) -> npt.NDArray[np.float32]:
        return calculate_tag_idf_weights(self.positive_movie_matrix)

    def find_referenced_movie_indices(self, query: str) -> list[int]:
        return find_referenced_movie_indices(
            query,
            self.title_candidates,
            min_single_word_title_length=Settings.MIN_SINGLE_WORD_TITLE_LENGTH,
        )

    def _calculate_reference_movie_similarities(
        self,
        user_vector: npt.NDArray[np.float32],
        reference_indices: list[int],
    ) -> npt.NDArray[np.float32]:
        reference_matrix = self.movie_matrix[reference_indices]
        reference_vector = reference_matrix.mean(axis=0).astype(np.float32)
        prompt_similarities = self._calculate_prompt_similarities(user_vector)
        reference_similarities = self._calculate_weighted_tversky_similarities(
            reference_vector
        )
        similarities = blend_reference_and_prompt_similarities(
            reference_similarities=reference_similarities,
            prompt_similarities=prompt_similarities,
            reference_weight=Settings.REFERENCE_MOVIE_VECTOR_WEIGHT,
            prompt_weight=Settings.PROMPT_VECTOR_WEIGHT_FOR_TITLE_QUERY,
        )
        return self._apply_reference_title_boost(
            similarities=similarities,
            reference_indices=reference_indices,
        )

    def _calculate_prompt_similarities(
        self, user_vector: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        return calculate_prompt_similarities(user_vector, self.movie_matrix)

    def _calculate_weighted_tversky_similarities(
        self,
        reference_vector: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        return calculate_weighted_tversky_similarities(
            reference_vector=reference_vector,
            positive_movie_matrix=self.positive_movie_matrix,
            tag_idf_weights=self.tag_idf_weights,
            label_positive_threshold=Settings.LABEL_POSITIVE_THRESHOLD,
            extra_tag_penalty=Settings.REFERENCE_EXTRA_TAG_PENALTY,
            missing_tag_penalty=Settings.REFERENCE_MISSING_TAG_PENALTY,
        )

    def _positive_movie_matrix(self) -> npt.NDArray[np.float32]:
        return self.positive_movie_matrix

    def _apply_reference_title_boost(
        self,
        similarities: npt.NDArray[np.float32],
        reference_indices: list[int],
    ) -> npt.NDArray[np.float32]:
        if not reference_indices:
            return similarities

        reference_tokens: set[str] = set()
        for idx in reference_indices:
            title = str(self.df.at[idx, "title"])
            reference_tokens.update(self._get_distinctive_title_tokens(title))

        if not reference_tokens:
            return similarities

        boosted_similarities = similarities.copy()
        for idx, title in self.df["title"].items():
            candidate_tokens = self._get_distinctive_title_tokens(str(title))
            if reference_tokens & candidate_tokens:
                boosted_similarities[int(idx)] += Settings.REFERENCE_TITLE_TOKEN_BOOST

        return np.minimum(boosted_similarities, 1.0).astype(np.float32)

    @staticmethod
    def _is_title_match_allowed(normalized_title: str, normalized_query: str) -> bool:
        return is_title_match_allowed(
            normalized_title,
            normalized_query,
            min_single_word_title_length=Settings.MIN_SINGLE_WORD_TITLE_LENGTH,
        )

    @staticmethod
    def _get_distinctive_title_tokens(title: str) -> set[str]:
        return get_distinctive_title_tokens(
            title,
            min_single_word_title_length=Settings.MIN_SINGLE_WORD_TITLE_LENGTH,
        )

    @staticmethod
    def normalize_for_matching(text: str) -> str:
        return normalize_title_for_matching(text)
