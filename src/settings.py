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
    RANDOM_SEED = 42
    TEST_SPLIT_SIZE = 0.1
    DEFAULT_TRAINING_EPOCHS = 12
    DEFAULT_TRAINING_BATCH_SIZE = 24
    DEFAULT_LEARNING_RATE = 2e-5
    DRY_RUN_TRAIN_SIZE = 20
    DRY_RUN_TEST_SIZE = 10
    WARMUP_STEPS = 0.1
    WEIGHT_DECAY = 0.01
    SAVE_TOTAL_LIMIT = 2
    METRIC_FOR_BEST_MODEL = "macro_f1"
    TRAINING_REPORT_TO = "tensorboard"
    THRESHOLD_SEARCH_START = 0.1
    THRESHOLD_SEARCH_STOP = 0.91
    THRESHOLD_SEARCH_STEP = 0.05

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
