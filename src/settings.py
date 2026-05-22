import os
from pathlib import Path

from dotenv import load_dotenv

from src.training.config import MODEL_NAME, MOOD_TAGS

load_dotenv()


class Settings:
    """Global object that contains projectwide constants."""

    BASE_DIR = Path(__file__).parent.parent
    DATASETS_DIR = Path(__file__).parent / "data"

    HF_ACCESS_TOKEN: str = os.getenv("HF_ACCESS_TOKEN", "")
    HF_REPO_ID: str = os.getenv("HF_REPO_ID", "")

    TELEGRAM_API_TOKEN: str | None = os.getenv("TELEGRAM_API_TOKEN")

    # Import config variables to expose them globally

    MODEL_NAME = MODEL_NAME
    MOOD_TAGS = MOOD_TAGS

    @classmethod
    def validate(cls) -> None:
        if not cls.TELEGRAM_API_TOKEN:
            raise ValueError(
                "TELEGRAM_API_TOKEN is missing. "
                "Please create a .env file based on .env.template."
            )
