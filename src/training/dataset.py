"""Module to load and preprocess the dataset for training."""

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import DatasetDict, load_dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from src.settings import Settings

SPLIT_NAMES = ("train", "validation", "test")


def load_movie_data(local_path: str | Path) -> DatasetDict:
    """Loads the movie-tag dataset and creates multi-label stratified splits.

    Args:
        local_path: Path to the local CSV file containing the dataset.

    Returns:
        A DatasetDict containing train, validation and held-out test splits.
    """
    dataset = load_dataset("csv", data_files=str(local_path))["train"]

    split_indices = build_multilabel_stratified_split_indices(
        labels=dataset["labels"],
        validation_size=Settings.VALIDATION_SPLIT_SIZE,
        test_size=Settings.TEST_SPLIT_SIZE,
        seed=Settings.RANDOM_SEED,
    )

    return DatasetDict({
        split_name: dataset.select(indices)
        for split_name, indices in split_indices.items()
    })


def build_multilabel_stratified_split_indices(
    *,
    labels: list[Any],
    validation_size: float,
    test_size: float,
    seed: int,
) -> dict[str, list[int]]:
    """Build deterministic split indices with iterative multi-label stratification."""

    label_matrix = np.asarray([_parse_label_vector(label) for label in labels])
    if label_matrix.ndim != 2:
        raise ValueError("Labels must be a 2-dimensional multi-label matrix.")

    total_rows = label_matrix.shape[0]
    target_sizes = _compute_target_split_sizes(
        total_rows=total_rows,
        validation_size=validation_size,
        test_size=test_size,
    )
    target_sizes_array = np.asarray(
        [target_sizes[split_name] for split_name in SPLIT_NAMES], dtype=np.int64
    )
    target_label_counts = (
        label_matrix.sum(axis=0, dtype=np.float64)
        * (target_sizes_array / total_rows)[:, np.newaxis]
    )

    rng = np.random.default_rng(seed)
    assigned_split = np.full(total_rows, fill_value=-1, dtype=np.int64)
    current_sizes = np.zeros(len(SPLIT_NAMES), dtype=np.int64)
    current_label_counts = np.zeros_like(target_label_counts)

    label_counts = label_matrix.sum(axis=0)
    for split_index in (SPLIT_NAMES.index("validation"), SPLIT_NAMES.index("test")):
        _seed_minimum_label_coverage(
            split_index=split_index,
            label_matrix=label_matrix,
            label_counts=label_counts,
            assigned_split=assigned_split,
            current_sizes=current_sizes,
            current_label_counts=current_label_counts,
            target_sizes=target_sizes_array,
            rng=rng,
        )

    label_order = [int(label_index) for label_index in np.argsort(label_counts)]
    for label_index in label_order:
        if label_counts[label_index] <= 0:
            continue

        label_indices = np.flatnonzero(label_matrix[:, label_index] > 0)
        rng.shuffle(label_indices)
        for row_index in label_indices:
            if assigned_split[row_index] != -1:
                continue
            split_index = _choose_split_for_row(
                row_index=int(row_index),
                label_matrix=label_matrix,
                target_sizes=target_sizes_array,
                target_label_counts=target_label_counts,
                current_sizes=current_sizes,
                current_label_counts=current_label_counts,
                rng=rng,
            )
            _assign_row_to_split(
                row_index=int(row_index),
                split_index=split_index,
                label_matrix=label_matrix,
                assigned_split=assigned_split,
                current_sizes=current_sizes,
                current_label_counts=current_label_counts,
            )

    remaining_indices = np.flatnonzero(assigned_split == -1)
    rng.shuffle(remaining_indices)
    for row_index in remaining_indices:
        split_index = _choose_split_for_row(
            row_index=int(row_index),
            label_matrix=label_matrix,
            target_sizes=target_sizes_array,
            target_label_counts=target_label_counts,
            current_sizes=current_sizes,
            current_label_counts=current_label_counts,
            rng=rng,
        )
        _assign_row_to_split(
            row_index=int(row_index),
            split_index=split_index,
            label_matrix=label_matrix,
            assigned_split=assigned_split,
            current_sizes=current_sizes,
            current_label_counts=current_label_counts,
        )

    return {
        split_name: sorted(
            int(row_index)
            for row_index in np.flatnonzero(assigned_split == split_index)
        )
        for split_index, split_name in enumerate(SPLIT_NAMES)
    }


def _parse_label_vector(label: Any) -> list[float]:
    if isinstance(label, str):
        return [float(value) for value in json.loads(label)]
    return [float(value) for value in label]


def _compute_target_split_sizes(
    *,
    total_rows: int,
    validation_size: float,
    test_size: float,
) -> dict[str, int]:
    holdout_size = validation_size + test_size
    if validation_size <= 0 or test_size <= 0 or holdout_size >= 1:
        raise ValueError(
            "VALIDATION_SPLIT_SIZE and TEST_SPLIT_SIZE must be positive and sum to "
            "less than 1."
        )

    split_fractions = {
        "train": 1 - holdout_size,
        "validation": validation_size,
        "test": test_size,
    }
    exact_sizes = {
        split_name: split_fraction * total_rows
        for split_name, split_fraction in split_fractions.items()
    }
    target_sizes = {
        split_name: math.floor(exact_sizes[split_name]) for split_name in SPLIT_NAMES
    }

    remaining_rows = total_rows - sum(target_sizes.values())
    split_order = sorted(
        SPLIT_NAMES,
        key=lambda split_name: exact_sizes[split_name] - target_sizes[split_name],
        reverse=True,
    )
    for split_name in split_order[:remaining_rows]:
        target_sizes[split_name] += 1

    return target_sizes


def _choose_split_for_row(
    *,
    row_index: int,
    label_matrix: np.ndarray[Any, Any],
    target_sizes: np.ndarray[Any, Any],
    target_label_counts: np.ndarray[Any, Any],
    current_sizes: np.ndarray[Any, Any],
    current_label_counts: np.ndarray[Any, Any],
    rng: np.random.Generator,
) -> int:
    available_splits = np.flatnonzero(current_sizes < target_sizes)
    if len(available_splits) == 0:
        available_splits = np.arange(len(SPLIT_NAMES))

    row_labels = label_matrix[row_index] > 0
    if row_labels.any():
        target_counts = np.maximum(target_label_counts[:, row_labels], 1.0)
        label_deficits = (
            target_label_counts[:, row_labels] - current_label_counts[:, row_labels]
        )
        label_scores = (label_deficits / target_counts).sum(axis=1)
    else:
        label_scores = np.zeros(len(SPLIT_NAMES), dtype=np.float64)

    size_scores = (target_sizes - current_sizes) / target_sizes
    return int(
        max(
            (int(split_index) for split_index in available_splits),
            key=lambda split_index: (
                float(label_scores[split_index]),
                float(size_scores[split_index]),
                float(rng.random()),
            ),
        )
    )


def _assign_row_to_split(
    *,
    row_index: int,
    split_index: int,
    label_matrix: np.ndarray[Any, Any],
    assigned_split: np.ndarray[Any, Any],
    current_sizes: np.ndarray[Any, Any],
    current_label_counts: np.ndarray[Any, Any],
) -> None:
    assigned_split[row_index] = split_index
    current_sizes[split_index] += 1
    current_label_counts[split_index] += label_matrix[row_index]


def _seed_minimum_label_coverage(
    *,
    split_index: int,
    label_matrix: np.ndarray[Any, Any],
    label_counts: np.ndarray[Any, Any],
    assigned_split: np.ndarray[Any, Any],
    current_sizes: np.ndarray[Any, Any],
    current_label_counts: np.ndarray[Any, Any],
    target_sizes: np.ndarray[Any, Any],
    rng: np.random.Generator,
) -> None:
    label_order = [int(label_index) for label_index in np.argsort(label_counts)]
    for label_index in label_order:
        if label_counts[label_index] <= 0:
            continue
        if current_label_counts[split_index, label_index] > 0:
            continue
        if current_sizes[split_index] >= target_sizes[split_index]:
            return

        candidate_indices = np.flatnonzero(
            (assigned_split == -1) & (label_matrix[:, label_index] > 0)
        )
        if len(candidate_indices) == 0:
            continue

        rng.shuffle(candidate_indices)
        missing_labels = current_label_counts[split_index] == 0
        row_index = int(
            max(
                (int(candidate_index) for candidate_index in candidate_indices),
                key=lambda candidate_index: int(
                    ((label_matrix[candidate_index] > 0) & missing_labels).sum()
                ),
            )
        )
        _assign_row_to_split(
            row_index=row_index,
            split_index=split_index,
            label_matrix=label_matrix,
            assigned_split=assigned_split,
            current_sizes=current_sizes,
            current_label_counts=current_label_counts,
        )


def get_tokenized_dataset(
    max_length: int = Settings.MAX_SEQUENCE_LENGTH,
    tokenizer_name_or_path: str | Path = Settings.MODEL_NAME,
) -> tuple[DatasetDict, Any]:
    """Loads the dataset, tokenizes it, and returns the tokenized DatasetDict and tokenizer.

    Args:
        max_length: Maximum sequence length for tokenization.
        tokenizer_name_or_path: Tokenizer name or local tokenizer directory.

    Returns:
        A tuple containing the tokenized DatasetDict and the tokenizer instance.
    """
    tokenizer: Any = AutoTokenizer.from_pretrained(str(tokenizer_name_or_path))  # type: ignore

    raw_dataset = load_movie_data(local_path=Settings.DATASET_PATH)

    def tokenize_and_format_fn(examples: dict[str, list[Any]]) -> dict[str, Any]:
        # Tokenize the movie overview texts
        tokenizer_out = cast(
            Any,
            tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
            ),
        )
        tokenized: dict[str, Any] = dict(tokenizer_out)  # type: ignore

        # Parse labels from string representation to float list if necessary
        labels_list: list[list[float]] = []
        for label in examples["labels"]:
            if isinstance(label, str):
                labels_list.append(json.loads(label))
            else:
                labels_list.append([float(val) for val in label])

        tokenized["labels"] = labels_list
        return tokenized

    print("Tokenize the dataset...")
    tokenized_dataset: DatasetDict = cast(Any, raw_dataset).map(
        tokenize_and_format_fn,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    return tokenized_dataset, cast(Any, tokenizer)
