import subprocess
from src.settings import Settings
from src.data.prepare_data import prepare_local_data
from src.data.hugging_face import upload_data_to_hugging_face


def main():
    print("Hello from waki-movies!")
    subprocess.run(["uvx", "pycowsay", "WakiKaki"])

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


if __name__ == "__main__":
    main()
