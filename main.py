"""
Main entry point for running the MoodPredictor and MovieRecommender.
"""

import asyncio

from src.data.prepare_data import create_local_data_file
from src.inference.inference import MoodPredictor
from src.inference.recommender import MovieRecommender
from src.settings import Settings
from src.telegram_bot.bot import TelegramBot
from src.training.train_model import run_training


def main() -> None:
    bot = TelegramBot()
    bot.start()

    # upload_path: Path = Settings.BASE_DIR / "src" / "data" / "raw-movie_data"
    # upload_data_to_hugging_face(path=upload_path, repo_id=Settings.HF_REPO_ID)

    local_csv_file = Settings.DATASET_PATH
    mood_tags_json_file = Settings.MOOD_TAGS_PATH
    local_models_dir = Settings.MODELS_DIR
    model_path = local_models_dir / "final_model"

    try:
        # Check if local CSV file exists
        if not local_csv_file.exists() or not mood_tags_json_file.exists():
            print("Preparing local data...")
            asyncio.run(create_local_data_file())

        # Checks if model exists
        if not model_path.exists():
            print("No trained model found. Start training...")
            print(
                "Run: "
                ""
                "uv run tensorboard --logdir src/training/models"
                ""
                "To monitor training progress in TensorBoard"
            )

            run_training(epochs=5, batch_size=8)

        # Init of predictor and recommender
        predictor = MoodPredictor()
        recommender = MovieRecommender(data_path=local_csv_file)

        prediction: dict[str, float] = predictor.predict(Settings.test_input)

        # Test if prediction works and print results
        print("Idea:")
        tag: str
        prob: float
        for tag, prob in prediction.items():
            print(f"{tag}: {prob:.4f}")

        # print("\nSuche nach den besten Filmen...")
        results = recommender.get_recommendations(prediction, top_k=3)

        for i, res in enumerate(results, 1):
            score_percent: float = float(res["similarity_score"]) * 100
            title: str = str(res["title"])
            overview: str = str(res["overview"])
            print(f"\nPlatz {i}: {title} (Match: {score_percent:.1f}%)")
            print(f"Beschreibung: {overview[:150]}...")

        bot.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
