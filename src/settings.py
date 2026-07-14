import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Global object that contains projectwide constants."""

    # API Tokens
    HF_ACCESS_TOKEN: str = os.getenv("HF_ACCESS_TOKEN", "")
    HF_REPO_ID: str | None = os.getenv("HF_REPO_ID") or "iaslfw/waki-movie_raw"
    HF_MODEL_ID: str | None = os.getenv("HF_MODEL_ID") or "iaslfw/waki-movie_model"
    TELEGRAM_API_TOKEN: str | None = os.getenv("TELEGRAM_API_TOKEN")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL") or "mistral-small-latest"
    MISTRAL_API_URL: str = os.getenv(
        "MISTRAL_API_URL",
        "https://api.mistral.ai/v1/chat/completions",
    )
    MISTRAL_TIMEOUT_SECONDS: float = float(os.getenv("MISTRAL_TIMEOUT_SECONDS", "10"))

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = Path(__file__).parent / "training" / "models"
    MOVIE_TAGS_PATH = Path(__file__).parent / "data" / "movie_tags.json"
    TAG_ALIAS_GROUPS_PATH = Path(__file__).parent / "data" / "tag_alias_groups.json"
    DATASET_PATH = Path(__file__).parent / "data" / "cleaned_movie_data.csv"
    DATA_REPORT_PATH = Path(__file__).parent / "data" / "data_quality_report.json"

    # Model and training configs
    MODEL_NAME = "distilbert-base-uncased"
    DECISION_THRESHOLD = 0.5
    MOVIE_TAG_COUNT = 150
    DEDUPLICATE_BY_TITLE = False
    TAG_SELECTION_THRESHOLD = 0.5
    LABEL_POSITIVE_THRESHOLD = 0.7
    MIN_POSITIVE_TAGS_PER_MOVIE = 2
    MAX_POSITIVE_TAGS_PER_MOVIE = 50
    MIN_OVERVIEW_WORDS = 12
    REFERENCE_MOVIE_VECTOR_WEIGHT = 0.7
    PROMPT_VECTOR_WEIGHT_FOR_TITLE_QUERY = 0.3
    REFERENCE_TITLE_TOKEN_BOOST = 0.25
    REFERENCE_EXTRA_TAG_PENALTY = 1.5
    REFERENCE_MISSING_TAG_PENALTY = 0.5
    MIN_SINGLE_WORD_TITLE_LENGTH = 5
    MAX_SEQUENCE_LENGTH = 512
    RANDOM_SEED = 42
    VALIDATION_SPLIT_SIZE = 0.1
    TEST_SPLIT_SIZE = 0.1
    DEFAULT_TRAINING_EPOCHS = 12
    DEFAULT_TRAINING_BATCH_SIZE = 24
    DEFAULT_LEARNING_RATE = 2e-5
    DRY_RUN_TRAIN_SIZE = 20
    DRY_RUN_VALIDATION_SIZE = 10
    DRY_RUN_TEST_SIZE = 10
    WARMUP_STEPS = 0.1
    WEIGHT_DECAY = 0.01
    SAVE_TOTAL_LIMIT = 2
    METRIC_FOR_BEST_MODEL = "roc_auc"
    TRAINING_REPORT_TO = "tensorboard"
    THRESHOLD_SEARCH_START = 0.1
    THRESHOLD_SEARCH_STOP = 0.91
    THRESHOLD_SEARCH_STEP = 0.05

    @classmethod
    def create_movie_tag_list(cls) -> list[str]:
        if cls.MOVIE_TAGS_PATH.exists():
            with open(cls.MOVIE_TAGS_PATH, "r", encoding="utf-8") as f:
                movie_tags = json.load(f)
                return movie_tags
        else:
            print(
                f"Warning: {cls.MOVIE_TAGS_PATH} not found. Returning empty tag list."
            )
            return []

    @classmethod
    def validate(cls) -> None:
        if not cls.TELEGRAM_API_TOKEN:
            raise ValueError(
                "TELEGRAM_API_TOKEN is missing. "
                "Please create a .env file based on .env.template."
            )
        if not cls.MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY is missing. "
                "WaKi-Movies uses Mistral for chat routing and localization."
            )

    @classmethod
    def require_dataset_source(cls) -> None:
        if not cls.HF_REPO_ID:
            raise ValueError(
                "HF_REPO_ID is missing. It is required to prepare local movie data "
                "when src/data/cleaned_movie_data.csv or src/data/movie_tags.json "
                "does not exist."
            )

    @classmethod
    def require_model_source(cls) -> None:
        if not cls.HF_MODEL_ID:
            raise ValueError(
                "HF_MODEL_ID is missing. It is required to download final_model "
                "when src/training/models/final_model does not exist."
            )
