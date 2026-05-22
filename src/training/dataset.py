"""Module to load and preprocess the dataset for training."""

import json
from typing import Any, cast

from datasets import DatasetDict, load_dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from src.settings import Settings
from src.training.config import MODEL_NAME


def load_movie_data(local_path: str | None = None) -> DatasetDict:
    """Loads the movie mood dataset from Hugging Face Hub or falls back to local CSV."""

    if Settings.HF_REPO_ID:
        try:
            print(f"Load Data from Hugging Face: {Settings.HF_REPO_ID}")

            dataset = load_dataset(Settings.HF_REPO_ID, token=Settings.HF_ACCESS_TOKEN)

            if "test" not in dataset:
                dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
            return dataset
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}. Trying local loading...")

    # Fallback to local file
    if local_path is None:
        local_path = str(
            Settings.DATASETS_DIR / "hf_folder" / "prepared_movie-data.csv"
        )

    print(f"Load dataset locally from: {local_path}")

    dataset = load_dataset("csv", data_files=local_path)

    dataset_dict = dataset["train"].train_test_split(test_size=0.1, seed=42)
    return dataset_dict


def get_tokenized_dataset(
    max_length: int = 512, local_path: str | None = None
) -> tuple[DatasetDict, Any]:
    """Loads the dataset, tokenizes it, and returns the tokenized DatasetDict and tokenizer."""
    tokenizer: Any = AutoTokenizer.from_pretrained(MODEL_NAME)  # type: ignore

    raw_dataset = load_movie_data(local_path=local_path)

    def tokenize_and_format_fn(examples: dict[str, list[Any]]) -> dict[str, Any]:
        # Tokenize the movie overview texts
        tokenizer_out = cast(
            Any,
            tokenizer(
                examples["text"],
                padding="max_length",
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
