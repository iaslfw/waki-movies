"""
Main entry point for running the MovieTagPredictor and MovieRecommender.
"""

import asyncio
import logging

from src.data.hugging_face import download_model_from_hugging_face
from src.data.prepare_data import create_local_data_file
from src.inference.inference import MovieTagPredictor
from src.inference.recommender import MovieRecommender
from src.settings import Settings
from src.telegram_bot.bot import TelegramBot

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()

    # upload_path: Path = Settings.BASE_DIR / "src" / "data" / "raw-movie_data"
    # upload_data_to_hugging_face(path=upload_path, repo_id=Settings.HF_REPO_ID)

    local_csv_file = Settings.DATASET_PATH
    movie_tags_json_file = Settings.MOVIE_TAGS_PATH
    local_models_dir = Settings.MODELS_DIR
    model_path = local_models_dir / "final_model"

    try:
        # Check if local CSV file exists
        if not local_csv_file.exists() or not movie_tags_json_file.exists():
            print("Preparing local data...")
            asyncio.run(create_local_data_file())

        # Checks if model exists
        if not model_path.exists():
            download_model_from_hugging_face(
                repo_id=Settings.HF_MODEL_ID, local_dir=model_path
            )

        # Init of predictor and recommender
        predictor = MovieTagPredictor()
        recommender = MovieRecommender(data_path=local_csv_file)

        # Bot
        bot = TelegramBot(predictor=predictor, recommender=recommender)
        bot.start()
        print("Bot started.")
        bot.wait()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
