"""
Recommender module for finding similar movies based on mood vectors.
"""

import ast
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from src.settings import Settings


class MovieRecommender:
    df: pd.DataFrame
    movie_matrix: npt.NDArray[np.float32]
    mood_tags: list[str]

    def __init__(self, data_path: str | Path) -> None:
        """
        Loads movie data and prepares the mood-vectors for recommendations

        Args:
            data_path: Path to the CSV file containing movie data.
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"No file found under: {path}.")

        self.mood_tags = Settings.create_mood_list()

        self.df = pd.read_csv(path)

        print("Create mood vectors...")

        def parse_labels(val: Any) -> list[float]:
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return [float(x) for x in parsed]  # type: ignore
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(val, list):
                return [float(x) for x in val]  # type: ignore
            return [0.0] * len(self.mood_tags)

        self.df["labels"] = self.df["labels"].apply(parse_labels)

        self.movie_matrix = np.array(self.df["labels"].tolist(), dtype=np.float32)

        print(f"Recommender ready! {len(self.df)} movies loaded.")

    def get_recommendations(
        self, user_vector_dict: dict[str, float], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Compares user vector with all movies, returns the top k most similiar movies.

        Args:
            user_vector_dict: A dictionary mapping mood tags to their corresponding values for the user.
            top_k: The number of top recommendations to return.
        """
        user_vector = np.array(
            [user_vector_dict.get(tag, 0.0) for tag in self.mood_tags], dtype=np.float32
        )

        user_vector_2d: npt.NDArray[np.float32] = user_vector.reshape(1, -1)

        raw_similarities = cosine_similarity(user_vector_2d, self.movie_matrix)[0]
        similarities: npt.NDArray[np.float32] = raw_similarities.astype(np.float32)

        top_indices: npt.NDArray[np.intp] = np.argsort(similarities)[::-1][:top_k]

        recommendations: list[dict[str, Any]] = []
        for idx_val in top_indices:
            idx = int(idx_val)

            title = str(self.df.at[idx, "title"])
            overview = str(self.df.at[idx, "text"])

            recommendations.append({
                "title": title,
                "overview": overview,
                "similarity_score": float(similarities[idx]),
            })

        return recommendations
