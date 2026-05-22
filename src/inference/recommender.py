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
    # Klasseneigenschaften deklarieren für Typsicherheit
    df: pd.DataFrame
    movie_matrix: npt.NDArray[np.float32]
    mood_tags: list[str]

    def __init__(self, data_path: str | Path) -> None:
        """
        Lädt die Filmdatenbank und bereitet die Vektor-Matrix vor.
        Wird beim Start des Servers/Bots nur einmal aufgerufen.
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Die Filmdatenbank unter {path} wurde nicht gefunden.")

        # Lade Mood-Tags dynamisch
        self.mood_tags = Settings.create_mood_list()

        self.df = pd.read_csv(path)

        print("Konvertiere Vektoren für schnelle Berechnungen...")
        
        # Typsicherer Parser für das Einlesen der String-Repräsentationen von Listen
        def parse_labels(val: Any) -> list[float]:
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                    if isinstance(parsed, list):
                        return [float(x) for x in parsed]
                except (ValueError, SyntaxError):
                    pass
            elif isinstance(val, list):
                return [float(x) for x in val]
            return [0.0] * len(self.mood_tags)

        self.df["labels"] = self.df["labels"].apply(parse_labels)

        # Typsichere Numpy-Matrix
        self.movie_matrix = np.array(self.df["labels"].tolist(), dtype=np.float32)

        print(f"Recommender bereit! {len(self.df)} Filme geladen.")

    def get_recommendations(
        self, user_vector_dict: dict[str, float], top_k: int = 3
    ) -> list[dict[str, Any]]:
        """
        Vergleicht den User-Vektor mit allen Filmen und gibt die Top K zurück.
        """
        # 1. Das Dictionary aus der inference.py in ein reines Numpy-Array umwandeln
        # Nutzt .get(tag, 0.0) für Robustheit, falls ein Tag im Dict fehlt
        user_vector = np.array(
            [user_vector_dict.get(tag, 0.0) for tag in self.mood_tags], 
            dtype=np.float32
        )

        # scikit-learn erwartet ein 2D-Array, daher formen wir (50,) zu (1, 50) um
        user_vector_2d: npt.NDArray[np.float32] = user_vector.reshape(1, -1)

        # 2. Kosinus-Ähnlichkeit berechnen
        raw_similarities = cosine_similarity(user_vector_2d, self.movie_matrix)[0]
        similarities: npt.NDArray[np.float32] = raw_similarities.astype(np.float32)

        # 3. Die besten Filme finden
        top_indices: npt.NDArray[np.intp] = np.argsort(similarities)[::-1][:top_k]

        # 4. Ergebnisse lesbar formatieren
        recommendations: list[dict[str, Any]] = []
        for idx_val in top_indices:
            idx = int(idx_val)
            # Typsichere Pandas Wertabfragen
            title = str(self.df.at[idx, "title"])
            overview = str(self.df.at[idx, "text"])
            
            recommendations.append({
                "title": title,
                "overview": overview,
                "similarity_score": float(similarities[idx]),
            })

        return recommendations
