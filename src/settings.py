import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Global object that contains projectwide constants."""

    # API Tokens
    HF_ACCESS_TOKEN: str = os.getenv("HF_ACCESS_TOKEN", "")
    HF_REPO_ID: str | None = os.getenv("HF_REPO_ID", None)
    HF_MODEL_ID: str | None = os.getenv("HF_MODEL_ID", None)
    TELEGRAM_API_TOKEN: str | None = os.getenv("TELEGRAM_API_TOKEN")

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATASETS_DIR = Path(__file__).parent / "data"
    MODELS_DIR = Path(__file__).parent / "training" / "models"
    MOOD_TAGS_PATH = Path(__file__).parent / "data" / "mood_tags.json"
    DATASET_PATH = Path(__file__).parent / "data" / "cleaned_movie_data.csv"

    # Model and training configs
    MODEL_NAME = "distilbert-base-uncased"
    DECISION_THRESHOLD = 0.5
    MAX_SEQUENCE_LENGTH = 512

    test_input = """I'm in the mood for a gripping survival story. 
    There's this movie about an astronaut who gets accidentally abandoned on Mars after 
    his crew assumes he died in a massive storm. It's all about his fight to stay alive 
    against the odds—it sounds like a really intense, high-stakes watch.
    """  # Description of aqua-man

    @classmethod
    def create_mood_list(cls) -> list[str]:
        if cls.MOOD_TAGS_PATH.exists():
            with open(cls.MOOD_TAGS_PATH, "r", encoding="utf-8") as f:
                mood_tags = json.load(f)
                return mood_tags
        else:
            print(
                f"Warning: {cls.MOOD_TAGS_PATH} not found. Returning empty mood list."
            )
            return []

    @classmethod
    def validate(cls) -> None:
        if not cls.TELEGRAM_API_TOKEN:
            raise ValueError(
                "TELEGRAM_API_TOKEN is missing. "
                "Please create a .env file based on .env.template."
            )
        if not cls.HF_ACCESS_TOKEN or not cls.HF_REPO_ID:
            print(
                "Warning: HF_ACCESS_TOKEN or HF_REPO_ID is missing. "
                "Hugging Face Hub upload will be disabled. Check .env.template"
            )
