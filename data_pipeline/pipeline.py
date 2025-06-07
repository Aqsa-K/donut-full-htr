from iiif_downloader import download_images_as_jpg
from annotations import generate_jsonl_file, generate_parquet_file_splits
from upload_to_hub import push_to_huggingface
import pandas as pd
import yaml


def run_pipeline(repo_id, train_parquet_path, val_parquet_path, test_parquet_path, dataset_infos_path, readme_path, gitattributes_path):
    # df = pd.read_csv("../data_preprocessing/original_data/1880_census_databasuttag.txt", sep="\t", dtype=str, encoding="latin1")
    # df = pd.read_csv("../data_preprocessing/output_data/sample_data_7512.csv")

    # download_images_as_jpg(df, "BILDNR" )  # Downloads to ./images/
    generate_jsonl_file()  # Reads ./images/, outputs ./annotations.jsonl
    # print("JSONL file generated")
    generate_parquet_file_splits()   # Reads annotations and saves ./output.parquet
    # print("Parquet file generated")
    push_to_huggingface(repo_id, train_parquet_path, val_parquet_path, test_parquet_path, dataset_infos_path, readme_path, gitattributes_path)  # Uploads to Hugging Face Hub
    print("Uploaded to Hugging Face Hub")


if __name__ == "__main__":
    # Load config from YAML
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Access values
    repo_id = config["repo_id"]
    train_parquet_path = config["train_parquet_path"]
    val_parquet_path = config["val_parquet_path"]
    test_parquet_path = config["test_parquet_path"]
    dataset_infos_path = config["dataset_infos_path"]
    readme_path = config["readme_path"]
    gitattributes_path = config["gitattributes_path"]
    

    # Example usage
    print("Repo ID:", repo_id)
    print("Train Parquet Path:", train_parquet_path)
    print("Validation Parquet Path:", val_parquet_path)
    print("Test Parquet Path:", test_parquet_path)
    print("Dataset Infos Path:", dataset_infos_path)
    print("Readme Path:", readme_path)
    print("Gitattributes Path:", gitattributes_path)

    run_pipeline(repo_id, train_parquet_path, val_parquet_path, test_parquet_path, dataset_infos_path, readme_path, gitattributes_path)