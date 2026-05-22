"""Module to prepare and merge TMDB & MovieLens data."""

import pandas as pd

from src.settings import Settings


def prepare_local_data() -> None:
    """Merges and prepares local data for upload in Hugging Face."""

    datasets_path = {
        "tmdb": Settings.DATASETS_DIR / "tmdb_movie_dataset_v11.csv",
        "links": Settings.DATASETS_DIR / "links.csv",
        "genome_scores": Settings.DATASETS_DIR / "genome-scores.csv",
        "genome_tags": Settings.DATASETS_DIR / "genome-tags.csv",
    }

    print("Lade TMDB-Daten...")

    tmdb_df = pd.read_csv(datasets_path["tmdb"])

    # Just keep movies with an overview (text)
    tmdb_df = tmdb_df[["id", "title", "overview"]].dropna(subset=["overview"])
    tmdb_df = tmdb_df.rename(columns={"id": "tmdbId"})

    print("Lade MovieLens Links...")
    links_df = pd.read_csv(datasets_path["links"])

    # Remove movies with empty tmdbId
    links_df = links_df.dropna(subset=["tmdbId"])
    links_df["tmdbId"] = links_df["tmdbId"].astype(int)

    # Connect TMDB data with MovieLens IDs
    movies_df = pd.merge(tmdb_df, links_df, on="tmdbId", how="inner")

    # Load movielens tag-genome (the "moods" of the movies)
    print("Lade Tag-Genome (Scores & Namen)...")
    scores_df = pd.read_csv(datasets_path["genome_scores"])
    tags_df = pd.read_csv(datasets_path["genome_tags"])

    # To create a focused model, only keep the top 50 most relevant tags (e.g. "dark comedy", "feel-good", "mind-bending")
    top_50_tag_ids = scores_df["tagId"].value_counts().head(50).index
    scores_filtered = scores_df[scores_df["tagId"].isin(top_50_tag_ids)]

    # Connect TagIds with Tag-Names
    scores_named = pd.merge(scores_filtered, tags_df, on="tagId")

    print("Erstelle die Multi-Label Matrix...")
    pivot_df = scores_named.pivot(index="movieId", columns="tag", values="relevance")
    pivot_df = (pivot_df >= 0.5).astype(float)

    # Merge vectors with movie texts
    print("Führe Texte und Vektoren zusammen...")
    final_df = pd.merge(movies_df, pivot_df, on="movieId", how="inner")

    tag_columns = pivot_df.columns.tolist()
    final_df["labels"] = final_df[tag_columns].values.tolist()
    final_df = final_df[["title", "overview", "labels"]]
    final_df = final_df.rename(columns={"overview": "text"})

    output_path = Settings.DATASETS_DIR / "hf_folder" / "prepared_movie-data.csv"
    final_df.to_csv(output_path, index=False)

    print(f"{len(final_df)} movies converted")
    print(f"{len(tag_columns)} Dimensions")
    print(f"File saved to: {output_path}")


if __name__ == "__main__":
    prepare_local_data()
