"""
Inference module for predicting movie tags.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.settings import Settings


class MovieTagPredictor:
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    model: PreTrainedModel
    device: torch.device
    movie_tags: list[str]

    def __init__(self, model_path: str | Path | None = None) -> None:
        """
        Load model and tokenizer into memory.

        Args:
            model_path: Optional path to the trained model directory.
            If None, it will look for the default path defined in Settings.
        """
        if model_path is None:
            model_path = (
                Settings.BASE_DIR / "src" / "training" / "models" / "final_model"
            )
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}.Please train the model first."
            )
        print(f"Loading model from {model_path}...")

        tokenizer_loaded = cast(Any, AutoTokenizer).from_pretrained(str(model_path))
        if not isinstance(
            tokenizer_loaded, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        ):
            raise TypeError("Wrong tokenizer type.")
        self.tokenizer = tokenizer_loaded

        model_loaded = cast(Any, AutoModelForSequenceClassification).from_pretrained(
            str(model_path)
        )
        if not isinstance(model_loaded, PreTrainedModel):
            raise TypeError("Wrong model type.")
        self.model = model_loaded

        # Either CUDA or Apple Silicon MPS or fallback to CPU
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(device=self.device)  # type: ignore
        self.model.eval()

        self.movie_tags = Settings.create_movie_tag_list()
        model_label_count = int(getattr(self.model.config, "num_labels", 0))
        if model_label_count != len(self.movie_tags):
            raise ValueError(
                "Model/tag mismatch: "
                f"model has {model_label_count} labels, "
                f"but {Settings.MOVIE_TAGS_PATH} contains {len(self.movie_tags)} tags. "
                "Regenerate the dataset and retrain the model."
            )

        print(
            f"Model loaded on device: {self.device}. {len(self.movie_tags)} movie tags loaded."
        )

    def predict(self, text: str) -> dict[str, float]:
        """
        Takes a text as input and returns a dictionary with all tags
        and their calculated probabilities.

        Args:
            text: The input text for which to predict movie tags.
        Returns:
            A dictionary mapping each movie tag to its predicted probability.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",  # 'pt' = PyTorch-Tensoren
            padding=True,
            truncation=True,
            max_length=Settings.MAX_SEQUENCE_LENGTH,
        )
        inputs_dict: dict[str, torch.Tensor] = dict(inputs)  # type: ignore
        inputs_device = {k: v.to(self.device) for k, v in inputs_dict.items()}

        with torch.no_grad():  # No gradients needed for inference
            outputs = self.model(**inputs_device)

        logits: npt.NDArray[Any] = outputs.logits[0].cpu().numpy()
        probs: npt.NDArray[Any] = 1.0 / (1.0 + np.exp(-logits))

        results: dict[str, float] = {}
        for i, tag in enumerate(self.movie_tags):
            results[tag] = float(probs[i])

        sorted_results = dict(
            sorted(results.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_results


MoodPredictor = MovieTagPredictor
