from datasets import Dataset
from huggingface_hub import HfApi, create_repo, upload_file
import pandas as pd
# from hf_transfer import upload_file


def push_to_huggingface(repo_id, parquet_path, dataset_infos_path, readme_path, gitattributes_path):
    # Create repo if it doesn't exist
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, private=True)
    except:
        pass  # Repo already exists

    # Load dataframe & convert to HuggingFace Dataset
    df = pd.read_parquet(parquet_path)
    hf_dataset = Dataset.from_pandas(df)

    # # Push to HF Hub
    # hf_dataset.push_to_hub(repo_id)

    upload_file(
    path_or_fileobj=parquet_path,
    path_in_repo="data/train.parquet",  # or just "train.parquet"
    repo_id=repo_id,
    repo_type="dataset"
    )

    # Upload dataset_infos.json
    upload_file(
        path_or_fileobj=dataset_infos_path,
        path_in_repo="dataset_infos.json",
        repo_id=repo_id,
        repo_type="dataset"
    )

    upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )

    upload_file(
        path_or_fileobj=gitattributes_path,
        path_in_repo=".gitattributes",
        repo_id=repo_id,
        repo_type="dataset"
    )

