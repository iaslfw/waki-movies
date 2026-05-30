"""Module to load and preprocess the dataset for training."""

import json
from pathlib import Path
from typing import Any, cast

from datasets import DatasetDict, load_dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from src.settings import Settings


def load_movie_data(local_path: str | Path) -> DatasetDict:
    """Loads the movie mood dataset from Hugging Face Hub or falls back to local CSV.

    Args:
        local_path: Path to the local CSV file containing the dataset.

    Returns:
        A DatasetDict containing the training and test splits.
    """
    dataset = load_dataset("csv", data_files=str(local_path))

    dataset_dict = dataset["train"].train_test_split(
        test_size=Settings.TEST_SPLIT_SIZE,
        seed=Settings.RANDOM_SEED,
    )
    return dataset_dict


def get_tokenized_dataset(
    max_length: int = Settings.MAX_SEQUENCE_LENGTH,
) -> tuple[DatasetDict, Any]:
    """Loads the dataset, tokenizes it, and returns the tokenized DatasetDict and tokenizer.

    Args:
        max_length: Maximum sequence length for tokenization.

    Returns:
        A tuple containing the tokenized DatasetDict and the tokenizer instance.
    """
    tokenizer: Any = AutoTokenizer.from_pretrained(Settings.MODEL_NAME)  # type: ignore

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
