from src.inference.inference import MoodPredictor
from src.settings import Settings
from src.training.hugging_face import upload_data_to_hugging_face
from src.training.prepare_data import prepare_local_data


def main() -> None:
    # run_bot()

    if (Settings.DATASETS_DIR / "hf_folder" / "prepared_movie-data.csv").exists():
        print("Datafile already exists.")
    else:
        prepare_local_data()

    if Settings.HF_ACCESS_TOKEN and Settings.HF_REPO_ID:
        print("Starting upload to Hugging Face Hub...")
        upload_data_to_hugging_face(
            path="./src/data/datasets/hf_folder",
            repo_id=Settings.HF_REPO_ID,
        )

    predictor = MoodPredictor()
    test_text = "Ein spannender Sci-Fi-Film mit einer düsteren Atmosphäre und überraschenden Wendungen."

    prediction = predictor.predict(test_text)

    print("Vorhersage für den Testtext:")
    for tag, prob in prediction.items():
        print(f"{tag}: {prob:.4f}")


if __name__ == "__main__":
    main()
