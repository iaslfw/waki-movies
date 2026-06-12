"""Module to prepare and merge TMDB & MovieLens data."""

import asyncio
import json
import re
from typing import cast

import pandas as pd  # type: ignore
from datasets import DatasetDict, disable_progress_bar, load_dataset  # type: ignore

from src.settings import Settings

disable_progress_bar()


def _normalize_tag(tag: object) -> str:
    tag_text = str(tag).strip().lower()
    tag_text = re.sub(r"\s+", " ", tag_text)
    return tag_text


def _load_tag_aliases() -> dict[str, str]:
    if not Settings.TAG_ALIAS_GROUPS_PATH.exists():
        return {}

    alias_groups = json.loads(
        Settings.TAG_ALIAS_GROUPS_PATH.read_text(encoding="utf-8")
    )
    aliases: dict[str, str] = {}
    for canonical_tag, equivalent_tags in alias_groups.items():
        canonical = _normalize_tag(canonical_tag)
        aliases[canonical] = canonical
        for equivalent_tag in equivalent_tags:
            aliases[_normalize_tag(equivalent_tag)] = canonical

    return aliases


def _apply_tag_alias(tag: object, aliases: dict[str, str]) -> str:
    normalized_tag = _normalize_tag(tag)
    return aliases.get(normalized_tag, normalized_tag)


def _normalize_title(title: object) -> str:
    title_text = str(title).strip().lower()
    return re.sub(r"\s+", " ", title_text)


def _normalize_text(text: object) -> str:
    text_value = str(text).strip().lower()
    return re.sub(r"\s+", " ", text_value)


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
        lambda: load_dataset(repo_id, data_files=file_name, token=token or None)
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

    report: dict[str, object] = {
        "settings": {
            "decision_threshold": Settings.DECISION_THRESHOLD,
            "movie_tag_count": Settings.MOVIE_TAG_COUNT,
            "deduplicate_by_title": Settings.DEDUPLICATE_BY_TITLE,
        },
        "input_rows": {
            "tmdb": int(len(tmdb_df)),
            "links": int(len(links_df)),
            "genome_scores": int(len(scores_df)),
            "genome_tags": int(len(tags_df)),
        },
    }
    tag_aliases = _load_tag_aliases()

    # Process TMDB movie data
    tmdb_missing_overview = int(tmdb_df["overview"].isna().sum())
    tmdb_df = tmdb_df[["id", "title", "overview"]].dropna(subset=["overview"])
    tmdb_df = tmdb_df.rename(columns={"id": "tmdbId"})
    tmdb_df = tmdb_df.drop_duplicates(subset=["tmdbId"], keep="first")

    # Process MovieLens links data
    links_missing_tmdb_id = int(links_df["tmdbId"].isna().sum())
    links_df = links_df.dropna(subset=["tmdbId"])
    links_df = links_df.astype({"tmdbId": int})
    links_df = links_df.drop_duplicates(subset=["movieId"], keep="first")
    links_df = links_df.drop_duplicates(subset=["tmdbId"], keep="first")

    # Merge TMDB data with MovieLens links to get the MovieLens IDs
    movies_df: pd.DataFrame = pd.merge(tmdb_df, links_df, on="tmdbId", how="inner")
    movies_before_dedupe = len(movies_df)
    movies_df = movies_df.drop_duplicates(subset=["movieId"], keep="first")
    movies_df = movies_df.drop_duplicates(subset=["tmdbId"], keep="first")
    movies_after_id_dedupe = len(movies_df)

    movies_df["normalized_title"] = movies_df["title"].map(_normalize_title)
    movies_df["normalized_overview"] = movies_df["overview"].map(_normalize_text)
    movies_before_exact_text_dedupe = len(movies_df)
    movies_df = movies_df.drop_duplicates(
        subset=["normalized_title", "normalized_overview"], keep="first"
    )
    movies_after_exact_text_dedupe = len(movies_df)
    duplicate_title_count = int(movies_df.duplicated(subset=["normalized_title"]).sum())
    if Settings.DEDUPLICATE_BY_TITLE:
        movies_df = movies_df.drop_duplicates(subset=["normalized_title"], keep="first")

    # Select the most frequent unique tags above the relevance threshold.
    tags_df = tags_df.copy()
    tags_df["tag"] = tags_df["tag"].map(lambda tag: _apply_tag_alias(tag, tag_aliases))
    tags_df = tags_df.drop_duplicates(subset=["tagId"], keep="first")
    top_tag_ids = (
        scores_df[scores_df["relevance"] >= Settings.DECISION_THRESHOLD]
        .merge(tags_df, on="tagId")
        .groupby("tag")["movieId"]
        .count()
        .sort_values(ascending=False)
        .head(Settings.MOVIE_TAG_COUNT)
        .index
        .tolist()
    )
    scores_named: pd.DataFrame = pd.merge(scores_df, tags_df, on="tagId")
    scores_filtered = scores_named[scores_named["tag"].isin(top_tag_ids)]

    # Create pivot table
    matrix_df: pd.DataFrame = scores_filtered.pivot_table(
        index="movieId", columns="tag", values="relevance", aggfunc="max"
    )
    matrix_df = (matrix_df >= Settings.DECISION_THRESHOLD).astype(float)

    tag_columns = matrix_df.columns.tolist()

    # Save selected movie tags directly in src/data.
    tags_output_path = Settings.MOVIE_TAGS_PATH
    tags_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_data = json.dumps(tag_columns, indent=4, ensure_ascii=False)
    with open(tags_output_path, "w", encoding="utf-8") as f:
        f.write(json_data)

    print(f"Selected movie tags saved to: {tags_output_path}")

    # Merge vectors with movie texts
    final_df: pd.DataFrame = pd.merge(movies_df, matrix_df, on="movieId", how="inner")
    final_label_distribution = final_df[tag_columns].sum(axis=0).sort_values(
        ascending=False
    )
    final_duplicate_title_count = int(
        final_df.duplicated(subset=["normalized_title"]).sum()
    )
    final_df["labels"] = final_df[tag_columns].values.tolist()
    final_df = final_df[["movieId", "tmdbId", "title", "overview", "labels"]]
    final_df = final_df.rename(columns={"overview": "text"})
    final_df = final_df.sort_values("movieId")

    output_path = Settings.DATASET_PATH

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    report.update({
        "filtering": {
            "tmdb_missing_overview_removed": tmdb_missing_overview,
            "links_missing_tmdb_id_removed": links_missing_tmdb_id,
            "movies_after_tmdb_links_merge": movies_before_dedupe,
            "movies_removed_by_id_dedupe": movies_before_dedupe - movies_after_id_dedupe,
            "movies_removed_by_exact_title_text_dedupe": (
                movies_before_exact_text_dedupe - movies_after_exact_text_dedupe
            ),
            "duplicate_titles_remaining_before_optional_title_dedupe": duplicate_title_count,
            "movies_after_optional_title_dedupe": int(len(movies_df)),
            "movies_after_genome_join": int(len(final_df)),
            "duplicate_titles_after_genome_join": final_duplicate_title_count,
        },
        "tags": {
            "selected_count": len(tag_columns),
            "selected_tags": tag_columns,
            "alias_groups_path": str(Settings.TAG_ALIAS_GROUPS_PATH),
            "alias_map": tag_aliases,
            "most_common_positive_labels": {
                tag: int(count)
                for tag, count in final_label_distribution.head(20).items()
            },
            "least_common_positive_labels": {
                tag: int(count)
                for tag, count in final_label_distribution.tail(20).items()
            },
        },
        "output": {
            "dataset_path": str(output_path),
            "tags_path": str(tags_output_path),
            "report_path": str(Settings.DATA_REPORT_PATH),
        },
    })

    Settings.DATA_REPORT_PATH.write_text(
        json.dumps(report, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    print(f"""
        {len(final_df)} movies converted, 
        with {len(tag_columns)} movie tags. 
        Prepared data saved to: {output_path}
        Data quality report saved to: {Settings.DATA_REPORT_PATH}

        Delete downloaded files with: 
        -> huggingface-cli delete-cache
    """)


if __name__ == "__main__":
    asyncio.run(create_local_data_file())
