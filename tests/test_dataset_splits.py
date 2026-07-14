import json

import numpy as np
import pytest

from src.training.dataset import (
    SPLIT_NAMES,
    build_multilabel_stratified_split_indices,
)


def test_stratified_split_indices_are_deterministic_complete_and_sized() -> None:
    base_labels = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ]
    labels = [
        json.dumps(label) if row_index % 2 == 0 else label
        for row_index, label in enumerate(base_labels * 2)
    ]

    first = build_multilabel_stratified_split_indices(
        labels=labels,
        validation_size=0.2,
        test_size=0.2,
        seed=123,
    )
    second = build_multilabel_stratified_split_indices(
        labels=labels,
        validation_size=0.2,
        test_size=0.2,
        seed=123,
    )

    assert first == second
    assert tuple(first) == SPLIT_NAMES
    assert {split_name: len(indices) for split_name, indices in first.items()} == {
        "train": 12,
        "validation": 4,
        "test": 4,
    }

    all_indices = [index for indices in first.values() for index in indices]
    assert sorted(all_indices) == list(range(len(labels)))
    assert len(all_indices) == len(set(all_indices))

    label_matrix = np.asarray([
        json.loads(label) if isinstance(label, str) else label for label in labels
    ])
    for split_name in SPLIT_NAMES:
        split_counts = label_matrix[first[split_name]].sum(axis=0)
        assert (split_counts > 0).all(), split_name


@pytest.mark.parametrize(
    ("validation_size", "test_size"),
    [
        (0.0, 0.2),
        (0.2, 0.0),
        (0.5, 0.5),
    ],
)
def test_stratified_split_rejects_invalid_holdout_sizes(
    validation_size: float,
    test_size: float,
) -> None:
    with pytest.raises(ValueError, match="must be positive and sum to less than 1"):
        build_multilabel_stratified_split_indices(
            labels=[[1, 0], [0, 1], [1, 1], [0, 0]],
            validation_size=validation_size,
            test_size=test_size,
            seed=1,
        )
