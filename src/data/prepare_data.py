"""Module to prepare and merge TMDB & MovieLens data."""

import asyncio
from typing import cast

import pandas as pd  # type: ignore
from datasets import DatasetDict, disable_progress_bar, load_dataset  # type: ignore

from src.data.preparation_steps import (
    build_data_quality_report,
    build_final_dataset,
    build_initial_report,
    build_label_matrix,
    load_tag_aliases,
    merge_movie_metadata,
    prepare_movie_links,
    prepare_tmdb_movies,
    select_top_tag_scores,
    write_json,
)
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

    report = build_initial_report(
        settings={
            "decision_threshold": Settings.DECISION_THRESHOLD,
            "tag_selection_threshold": Settings.TAG_SELECTION_THRESHOLD,
            "label_positive_threshold": Settings.LABEL_POSITIVE_THRESHOLD,
            "movie_tag_count": Settings.MOVIE_TAG_COUNT,
            "deduplicate_by_title": Settings.DEDUPLICATE_BY_TITLE,
            "min_positive_tags_per_movie": Settings.MIN_POSITIVE_TAGS_PER_MOVIE,
            "max_positive_tags_per_movie": Settings.MAX_POSITIVE_TAGS_PER_MOVIE,
            "min_overview_words": Settings.MIN_OVERVIEW_WORDS,
        },
        tmdb_df=tmdb_df,
        links_df=links_df,
        scores_df=scores_df,
        tags_df=tags_df,
    )
    tag_aliases = load_tag_aliases(Settings.TAG_ALIAS_GROUPS_PATH)

    tmdb_movies = prepare_tmdb_movies(tmdb_df, Settings.MIN_OVERVIEW_WORDS)
    movie_links = prepare_movie_links(links_df)
    movie_metadata = merge_movie_metadata(
        tmdb_movies.df,
        movie_links.df,
        deduplicate_by_title=Settings.DEDUPLICATE_BY_TITLE,
    )

    selected_tag_scores = select_top_tag_scores(
        scores_df=scores_df,
        tags_df=tags_df,
        tag_aliases=tag_aliases,
        tag_selection_threshold=Settings.TAG_SELECTION_THRESHOLD,
        movie_tag_count=Settings.MOVIE_TAG_COUNT,
    )
    label_matrix = build_label_matrix(
        selected_tag_scores.scores_filtered,
        label_positive_threshold=Settings.LABEL_POSITIVE_THRESHOLD,
        min_positive_tags_per_movie=Settings.MIN_POSITIVE_TAGS_PER_MOVIE,
        max_positive_tags_per_movie=Settings.MAX_POSITIVE_TAGS_PER_MOVIE,
    )
    tag_columns = label_matrix.tag_columns

    # Save selected movie tags directly in src/data.
    tags_output_path = Settings.MOVIE_TAGS_PATH
    write_json(tags_output_path, tag_columns)

    print(f"Selected movie tags saved to: {tags_output_path}")

    # Merge vectors with movie texts
    final_dataset = build_final_dataset(
        movie_metadata.df,
        label_matrix.df,
        tag_columns,
    )

    output_path = Settings.DATASET_PATH

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_dataset.df.to_csv(output_path, index=False)

    report = build_data_quality_report(
        base_report=report,
        tmdb_movies=tmdb_movies,
        movie_links=movie_links,
        movie_metadata=movie_metadata,
        label_matrix=label_matrix,
        final_dataset=final_dataset,
        tag_aliases=tag_aliases,
        tag_alias_groups_path=Settings.TAG_ALIAS_GROUPS_PATH,
        output_path=output_path,
        tags_output_path=tags_output_path,
        report_path=Settings.DATA_REPORT_PATH,
    )
    write_json(Settings.DATA_REPORT_PATH, report)

    print(f"""
        {len(final_dataset.df)} movies converted, 
        with {len(tag_columns)} movie tags. 
        Prepared data saved to: {output_path}
        Data quality report saved to: {Settings.DATA_REPORT_PATH}

        Delete downloaded files with: 
        -> huggingface-cli delete-cache
    """)


if __name__ == "__main__":
    asyncio.run(create_local_data_file())
