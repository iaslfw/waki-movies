"""
Inference module for predicting movie mood tags.
"""

from pathlib import Path

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


class MoodPredictor:
    # Explizite Typendeklarationen auf Klassenebene
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
    model: PreTrainedModel
    device: torch.device
    mood_tags: list[str]

    def __init__(self, model_path: str | Path | None = None) -> None:
        """
        Lädt das Modell und den Tokenizer in den Arbeitsspeicher.
        Wird beim Start des Servers/Bots genau einmal aufgerufen.
        """
        if model_path is None:
            model_path = (
                Settings.BASE_DIR / "src" / "training" / "models" / "final_model"
            )
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Das trainierte Modell wurde unter {model_path} nicht gefunden. "
                "Bitte führe zuerst das Training aus."
            )
        print(f"Lade Mood-Predictor von {model_path}...")
        # Typsicheres Laden des Tokenizers
        tokenizer_loaded = AutoTokenizer.from_pretrained(str(model_path))
        if not isinstance(
            tokenizer_loaded, (PreTrainedTokenizer, PreTrainedTokenizerFast)
        ):
            raise TypeError("Geladener Tokenizer entspricht nicht dem erwarteten Typ.")
        self.tokenizer = tokenizer_loaded

        model_loaded = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        )
        if not isinstance(model_loaded, PreTrainedModel):
            raise TypeError("Geladenes Modell entspricht nicht dem erwarteten Typ.")
        self.model = model_loaded

        # Device-Handling (Nutzt CUDA oder Apple Silicon MPS, falls verfügbar)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        # Setzt das Modell in den Evaluierungs-Modus (schaltet Dropout etc. ab)
        self.model.eval()

        # Lade die Mood-Tags dynamisch
        self.mood_tags = Settings.create_mood_list()
        print(
            f"Modell auf Gerät geladen: {self.device}. {len(self.mood_tags)} Mood-Tags geladen."
        )

    def predict(self, text: str) -> dict[str, float]:
        """
        Nimmt einen Text entgegen und gibt ein Dictionary mit allen Tags
        und ihren berechneten Wahrscheinlichkeiten zurück.
        """
        # 1. Text tokenisieren und Tensors auf das richtige Gerät verschieben
        inputs = self.tokenizer(
            text,
            return_tensors="pt",  # 'pt' steht für PyTorch-Tensoren
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 2. Modellvorhersage (ohne Gradientenberechnung -> spart Speicher & Zeit)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Logits auf CPU kopieren, extrahieren und Sigmoid anwenden (Werte zwischen 0 und 1)
        logits: npt.NDArray[np.float32] = outputs.logits[0].cpu().numpy()
        probs: npt.NDArray[np.float32] = 1.0 / (1.0 + np.exp(-logits))

        # 4. Den Vektor (probs) mit den Namen der Mood-Tags verknüpfen
        results: dict[str, float] = {}
        for i, tag in enumerate(self.mood_tags):
            results[tag] = float(probs[i])

        # 5. Optional: Zur besseren Übersicht absteigend sortieren
        sorted_results = dict(
            sorted(results.items(), key=lambda item: item[1], reverse=True)
        )
        return sorted_results
