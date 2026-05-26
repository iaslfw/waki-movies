"""Module to upload merged dataset to HuggingFace Hub."""

from pathlib import Path

from huggingface_hub import login, upload_folder
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.settings import Settings


def upload_data_to_hugging_face(path: Path, repo_id: str) -> None:
    """Example function to connect to Hugging Face Hub and upload a dataset.

    Args:
        path: Path to the folder containing the dataset.
        repo_id: Repository ID in the format "username/repo_name" where the dataset should be uploaded.
    """

    login(token=Settings.HF_ACCESS_TOKEN)

    try:
        upload_folder(folder_path=path, repo_id=repo_id, repo_type="dataset")
        print(f"Dataset successfully uploaded to Hugging Face Hub: {repo_id}")
    except Exception as e:
        print(f"Error occurred while uploading data to Hugging Face Hub: {e}")


def download_model_from_hugging_face(
    repo_id: str | None, local_dir: Path = Settings.MODELS_DIR
) -> None:
    """Downloads model from Hugging Face Hub and stores it locally.

    Args:
        repo_id: Repository ID for the model in format "username/repo_name".
        local_dir: Local directory where model should be saved. Defaults to MODELS_DIR setting.

    Returns:
        None
    """

    try:
        print(
            ""
            "No trained model found...\n"
            "Downloading pre-trained model from HuggingFace\n"
            "+------------------------------+\n"
            "Want to train own model?\n"
            "Run: uv run -m src.training.train_model\n"
            "+------------------------------+\n"
        )

        model = AutoModelForSequenceClassification.from_pretrained(  # type: ignore
            repo_id,  # type: ignore
        )
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore
            repo_id,  # type: ignore
        )

        model.save_pretrained(local_dir)  # type: ignore
        tokenizer.save_pretrained(local_dir)  # type: ignore

        print(f"Model successfully downloaded and saved to\n-> {local_dir}")
    except Exception as e:
        print(f"Error occurred while downloading model from Hugging Face Hub: {e}")
