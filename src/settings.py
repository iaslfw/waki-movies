import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Global object that contains projectwide constants."""

    BASE_DIR = Path(__file__).parent.parent
    DATASETS_DIR = Path(__file__).parent / "data" / "datasets"

    HF_ACCESS_TOKEN: str = os.getenv("HF_ACCESS_TOKEN", "")
    HF_REPO_ID: str = os.getenv("HF_REPO_ID", "")
