"""Module to prepare and merge TMDB & MovieLens data."""

import asyncio
import json
from typing import cast

import pandas as pd  # type: ignore
from datasets import DatasetDict, disable_progress_bar, load_dataset  # type: ignore

from src.settings import Settings

disable_progress_bar()


async def _download_data_from_hugging_face(
    repo_id: str, file_name: str, token: str
) -> pd.DataFrame:
    """Loads a specific dataset file from Hugging Face asynchronously on a background thread.

    Args:
        repo_id: Repository ID in the format "username/repo_name" where the dataset is located.
        file_name: Name of the file to load from the dataset repository.
        token: Hugging Face access token for authentication.
    """
    print(f"Start Download: {file_name}...")
    dataset: DatasetDict = await asyncio.to_thread(
        lambda: load_dataset(repo_id, data_files=file_name, token=token)
    )

    df = cast(pd.DataFrame, dataset["train"].to_pandas())

    return df


async def create_local_data_file() -> None:
    """Merges and prepares local data for upload in Hugging Face."""

    repo_id = Settings.HF_REPO_ID
    token = Settings.HF_ACCESS_TOKEN

    print(f"Starting asynchronous downloads from HF-Repository: {repo_id}")

    # Start all 4 downloads concurrently
    tmdb_task = _download_data_from_hugging_face(
        repo_id, "tmdb_movie_dataset_v11.csv", token
    )
    links_task = _download_data_from_hugging_face(repo_id, "links.csv", token)
    scores_task = _download_data_from_hugging_face(repo_id, "genome-scores.csv", token)
    tags_task = _download_data_from_hugging_face(repo_id, "genome-tags.csv", token)

    # Wait for all downloads to finish concurrently
    tmdb_df, links_df, scores_df, tags_df = await asyncio.gather(
        tmdb_task, links_task, scores_task, tags_task
    )

    print("All downloads finished. Starting processing data...")

    # Process TMDB movie data
    tmdb_df = tmdb_df[["id", "title", "overview"]].dropna(subset=["overview"])
    tmdb_df = tmdb_df.rename(columns={"id": "tmdbId"})

    # Process MovieLens links data
    links_df = links_df.dropna(subset=["tmdbId"])
    links_df = links_df.astype({"tmdbId": int})

    # Merge TMDB data with MovieLens links to get the MovieLens IDs
    movies_df: pd.DataFrame = pd.merge(tmdb_df, links_df, on="tmdbId", how="inner")

    # Keeping only most relevant tags (100)
    top_tag_ids = (
        scores_df[scores_df["relevance"] >= Settings.DECISION_THRESHOLD]["tagId"]
        .value_counts()
        .head(100)
        .index
    )
    scores_filtered = scores_df[scores_df["tagId"].isin(top_tag_ids)]
    scores_named: pd.DataFrame = pd.merge(scores_filtered, tags_df, on="tagId")

    # Create pivot table
    matrix_df: pd.DataFrame = scores_named.pivot(
        index="movieId", columns="tag", values="relevance"
    )
    matrix_df = (matrix_df >= Settings.DECISION_THRESHOLD).astype(float)

    tag_columns = matrix_df.columns.tolist()

    # Save the selected mood tags to mood_tags.json directly in src/data
    tags_output_path = Settings.MOOD_TAGS_PATH
    tags_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = json.dumps(tag_columns, indent=4, ensure_ascii=False)
    with open(tags_output_path, "w", encoding="utf-8") as f:
        f.write(json_data)

    print(f"Selected mood tags saved to: {tags_output_path}")

    # Merge vectors with movie texts
    final_df: pd.DataFrame = pd.merge(movies_df, matrix_df, on="movieId", how="inner")
    final_df["labels"] = final_df[tag_columns].values.tolist()
    final_df = final_df[["title", "overview", "labels"]]
    final_df = final_df.rename(columns={"overview": "text"})

    output_path = Settings.DATASET_PATH

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"""
        {len(final_df)} movies converted, 
        with {len(tag_columns)} mood tags. 
        Prepared data saved to: {output_path}

        Delete downloaded files with: 
        -> huggingface-cli delete-cache
    """)


if __name__ == "__main__":
    asyncio.run(create_local_data_file())
