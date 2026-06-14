"""Pure processing steps for building the local movie-tag dataset."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore


@dataclass(frozen=True)
class TmdbMovies:
    df: pd.DataFrame
    missing_overview_count: int
    too_short_overview_count: int


@dataclass(frozen=True)
class MovieLinks:
    df: pd.DataFrame
    missing_tmdb_id_count: int


@dataclass(frozen=True)
class MovieMetadata:
    df: pd.DataFrame
    before_id_dedupe_count: int
    after_id_dedupe_count: int
    before_exact_text_dedupe_count: int
    after_exact_text_dedupe_count: int
    duplicate_title_count: int


@dataclass(frozen=True)
class SelectedTagScores:
    scores_filtered: pd.DataFrame


@dataclass(frozen=True)
class LabelMatrix:
    df: pd.DataFrame
    tag_columns: list[str]
    before_density_filter_count: int
    sparse_movie_count: int
    dense_movie_count: int


@dataclass(frozen=True)
class FinalDataset:
    df: pd.DataFrame
    label_distribution: pd.Series
    positive_label_counts: pd.Series
    duplicate_title_count: int


def normalize_tag(tag: object) -> str:
    tag_text = str(tag).strip().lower()
    return re.sub(r"\s+", " ", tag_text)


def load_tag_aliases(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    alias_groups = json.loads(path.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for canonical_tag, equivalent_tags in alias_groups.items():
        canonical = normalize_tag(canonical_tag)
        aliases[canonical] = canonical
        for equivalent_tag in equivalent_tags:
            aliases[normalize_tag(equivalent_tag)] = canonical

    return aliases


def apply_tag_alias(tag: object, aliases: dict[str, str]) -> str:
    normalized_tag = normalize_tag(tag)
    return aliases.get(normalized_tag, normalized_tag)


def normalize_title(title: object) -> str:
    title_text = str(title).strip().lower()
    return re.sub(r"\s+", " ", title_text)


def normalize_text(text: object) -> str:
    text_value = str(text).strip().lower()
    return re.sub(r"\s+", " ", text_value)


def has_min_word_count(text: object, min_words: int) -> bool:
    return len(str(text).split()) >= min_words


def build_initial_report(
    *,
    settings: dict[str, object],
    tmdb_df: pd.DataFrame,
    links_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    tags_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        "settings": settings,
        "input_rows": {
            "tmdb": int(len(tmdb_df)),
            "links": int(len(links_df)),
            "genome_scores": int(len(scores_df)),
            "genome_tags": int(len(tags_df)),
        },
    }


def prepare_tmdb_movies(tmdb_df: pd.DataFrame, min_overview_words: int) -> TmdbMovies:
    missing_overview_count = int(tmdb_df["overview"].isna().sum())
    prepared_df = tmdb_df[["id", "title", "overview"]].dropna(subset=["overview"])
    overview_has_min_words = prepared_df["overview"].map(
        lambda overview: has_min_word_count(overview, min_overview_words)
    )
    too_short_overview_count = int((~overview_has_min_words).sum())
    prepared_df = prepared_df[overview_has_min_words]
    prepared_df = prepared_df.rename(columns={"id": "tmdbId"})
    prepared_df = prepared_df.drop_duplicates(subset=["tmdbId"], keep="first")

    return TmdbMovies(
        df=prepared_df,
        missing_overview_count=missing_overview_count,
        too_short_overview_count=too_short_overview_count,
    )


def prepare_movie_links(links_df: pd.DataFrame) -> MovieLinks:
    missing_tmdb_id_count = int(links_df["tmdbId"].isna().sum())
    prepared_df = links_df.dropna(subset=["tmdbId"])
    prepared_df = prepared_df.astype({"tmdbId": int})
    prepared_df = prepared_df.drop_duplicates(subset=["movieId"], keep="first")
    prepared_df = prepared_df.drop_duplicates(subset=["tmdbId"], keep="first")

    return MovieLinks(df=prepared_df, missing_tmdb_id_count=missing_tmdb_id_count)


def merge_movie_metadata(
    tmdb_df: pd.DataFrame,
    links_df: pd.DataFrame,
    *,
    deduplicate_by_title: bool,
) -> MovieMetadata:
    movies_df: pd.DataFrame = pd.merge(tmdb_df, links_df, on="tmdbId", how="inner")
    before_id_dedupe_count = len(movies_df)
    movies_df = movies_df.drop_duplicates(subset=["movieId"], keep="first")
    movies_df = movies_df.drop_duplicates(subset=["tmdbId"], keep="first")
    after_id_dedupe_count = len(movies_df)

    movies_df["normalized_title"] = movies_df["title"].map(normalize_title)
    movies_df["normalized_overview"] = movies_df["overview"].map(normalize_text)
    before_exact_text_dedupe_count = len(movies_df)
    movies_df = movies_df.drop_duplicates(
        subset=["normalized_title", "normalized_overview"], keep="first"
    )
    after_exact_text_dedupe_count = len(movies_df)
    duplicate_title_count = int(movies_df.duplicated(subset=["normalized_title"]).sum())
    if deduplicate_by_title:
        movies_df = movies_df.drop_duplicates(subset=["normalized_title"], keep="first")

    return MovieMetadata(
        df=movies_df,
        before_id_dedupe_count=before_id_dedupe_count,
        after_id_dedupe_count=after_id_dedupe_count,
        before_exact_text_dedupe_count=before_exact_text_dedupe_count,
        after_exact_text_dedupe_count=after_exact_text_dedupe_count,
        duplicate_title_count=duplicate_title_count,
    )


def select_top_tag_scores(
    *,
    scores_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    tag_aliases: dict[str, str],
    tag_selection_threshold: float,
    movie_tag_count: int,
) -> SelectedTagScores:
    normalized_tags_df = tags_df.copy()
    normalized_tags_df["tag"] = normalized_tags_df["tag"].map(
        lambda tag: apply_tag_alias(tag, tag_aliases)
    )
    normalized_tags_df = normalized_tags_df.drop_duplicates(
        subset=["tagId"], keep="first"
    )
    top_tags = (
        scores_df[scores_df["relevance"] >= tag_selection_threshold]
        .merge(normalized_tags_df, on="tagId")
        .groupby("tag")["movieId"]
        .count()
        .sort_values(ascending=False)
        .head(movie_tag_count)
        .index.tolist()
    )
    scores_named: pd.DataFrame = pd.merge(scores_df, normalized_tags_df, on="tagId")
    scores_filtered = scores_named[scores_named["tag"].isin(top_tags)]

    return SelectedTagScores(scores_filtered=scores_filtered)


def build_label_matrix(
    scores_filtered: pd.DataFrame,
    *,
    label_positive_threshold: float,
    min_positive_tags_per_movie: int,
    max_positive_tags_per_movie: int,
) -> LabelMatrix:
    matrix_df: pd.DataFrame = scores_filtered.pivot_table(
        index="movieId", columns="tag", values="relevance", aggfunc="max"
    )
    matrix_df = (matrix_df >= label_positive_threshold).astype(float)
    positive_label_counts = matrix_df.sum(axis=1)
    before_density_filter_count = len(matrix_df)
    sparse_movie_count = int(
        (positive_label_counts < min_positive_tags_per_movie).sum()
    )
    dense_movie_count = int((positive_label_counts > max_positive_tags_per_movie).sum())
    matrix_df = matrix_df[
        (positive_label_counts >= min_positive_tags_per_movie)
        & (positive_label_counts <= max_positive_tags_per_movie)
    ]

    return LabelMatrix(
        df=matrix_df,
        tag_columns=matrix_df.columns.tolist(),
        before_density_filter_count=before_density_filter_count,
        sparse_movie_count=sparse_movie_count,
        dense_movie_count=dense_movie_count,
    )


def build_final_dataset(
    movies_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    tag_columns: list[str],
) -> FinalDataset:
    final_df: pd.DataFrame = pd.merge(movies_df, matrix_df, on="movieId", how="inner")
    label_distribution = final_df[tag_columns].sum(axis=0).sort_values(ascending=False)
    positive_label_counts = final_df[tag_columns].sum(axis=1)
    duplicate_title_count = int(final_df.duplicated(subset=["normalized_title"]).sum())
    final_df["labels"] = final_df[tag_columns].values.tolist()
    final_df = final_df[["movieId", "tmdbId", "title", "overview", "labels"]]
    final_df = final_df.rename(columns={"overview": "text"})
    final_df = final_df.sort_values("movieId")

    return FinalDataset(
        df=final_df,
        label_distribution=label_distribution,
        positive_label_counts=positive_label_counts,
        duplicate_title_count=duplicate_title_count,
    )


def build_data_quality_report(
    *,
    base_report: dict[str, object],
    tmdb_movies: TmdbMovies,
    movie_links: MovieLinks,
    movie_metadata: MovieMetadata,
    label_matrix: LabelMatrix,
    final_dataset: FinalDataset,
    tag_aliases: dict[str, str],
    tag_alias_groups_path: Path,
    output_path: Path,
    tags_output_path: Path,
    report_path: Path,
) -> dict[str, object]:
    report = dict(base_report)
    final_positive_label_counts = final_dataset.positive_label_counts

    report.update({
        "filtering": {
            "tmdb_missing_overview_removed": tmdb_movies.missing_overview_count,
            "tmdb_too_short_overview_removed": tmdb_movies.too_short_overview_count,
            "links_missing_tmdb_id_removed": movie_links.missing_tmdb_id_count,
            "movies_after_tmdb_links_merge": movie_metadata.before_id_dedupe_count,
            "movies_removed_by_id_dedupe": (
                movie_metadata.before_id_dedupe_count
                - movie_metadata.after_id_dedupe_count
            ),
            "movies_removed_by_exact_title_text_dedupe": (
                movie_metadata.before_exact_text_dedupe_count
                - movie_metadata.after_exact_text_dedupe_count
            ),
            "duplicate_titles_remaining_before_optional_title_dedupe": (
                movie_metadata.duplicate_title_count
            ),
            "movies_after_optional_title_dedupe": int(len(movie_metadata.df)),
            "movies_with_genome_vectors_before_label_density_filter": int(
                label_matrix.before_density_filter_count
            ),
            "movies_removed_by_too_few_positive_labels": (
                label_matrix.sparse_movie_count
            ),
            "movies_removed_by_too_many_positive_labels": (
                label_matrix.dense_movie_count
            ),
            "movies_after_genome_join": int(len(final_dataset.df)),
            "duplicate_titles_after_genome_join": final_dataset.duplicate_title_count,
        },
        "label_density": {
            "min_positive_labels_per_movie": int(final_positive_label_counts.min()),
            "median_positive_labels_per_movie": float(
                final_positive_label_counts.median()
            ),
            "p90_positive_labels_per_movie": float(
                final_positive_label_counts.quantile(0.90)
            ),
            "p95_positive_labels_per_movie": float(
                final_positive_label_counts.quantile(0.95)
            ),
            "p99_positive_labels_per_movie": float(
                final_positive_label_counts.quantile(0.99)
            ),
            "max_positive_labels_per_movie": int(final_positive_label_counts.max()),
        },
        "tags": {
            "selected_count": len(label_matrix.tag_columns),
            "selected_tags": label_matrix.tag_columns,
            "alias_groups_path": str(tag_alias_groups_path),
            "alias_map": tag_aliases,
            "most_common_positive_labels": {
                tag: int(count)
                for tag, count in final_dataset.label_distribution.head(20).items()
            },
            "least_common_positive_labels": {
                tag: int(count)
                for tag, count in final_dataset.label_distribution.tail(20).items()
            },
        },
        "output": {
            "dataset_path": str(output_path),
            "tags_path": str(tags_output_path),
            "report_path": str(report_path),
        },
    })

    return report


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
