"""Module to upload merged dataset to HuggingFace Hub."""

from huggingface_hub import login, upload_folder

from src.settings import Settings


def upload_data_to_hugging_face(path: str, repo_id: str) -> None:
    """Example function to connect to Hugging Face Hub and upload a dataset."""

    login(token=Settings.HF_ACCESS_TOKEN)

    try:
        upload_folder(folder_path=path, repo_id=repo_id, repo_type="dataset")
        print(f"Dataset successfully uploaded to Hugging Face Hub: {repo_id}")
    except Exception as e:
        print(f"Error occurred while uploading data to Hugging Face Hub: {e}")
