"""
This script provides utility functions for inspecting, processing, and displaying datasets 
related to handwritten text recognition (HTR). It includes functionality for reading and 
inspecting parquet files, generating examples from JSONL files, and displaying dataset 
information from the Hugging Face datasets library.
Functions:
- read_and_inspect_parquet(file_path): Reads a parquet file into a pandas DataFrame, 
    inspects its structure, and prints details about specific columns.
- generate_examples(data_dir): Generates examples from a JSONL file located in the 
    specified directory, yielding parsed ground truth data and image paths.
- inspect_data_examples(data_dir): Inspects and prints details of examples generated 
    from a JSONL file, including parsed data and image paths.
- display_dataset_info(dataset_name): Loads a dataset from the Hugging Face datasets 
    library, prints its structure and features, and displays an example image.
The script also includes example usage of these functions to read local parquet files 
and verify uploaded datasets.
"""

import json
import os
import pandas as pd
from datasets import load_dataset


def read_and_inspect_parquet(file_path):
    df = pd.read_parquet(file_path)
    print(df.head())
    print(df.iloc[0]['ground_truth'])
    print(type(df.iloc[0]['ground_truth']))
    print(type(df.iloc[0]['image']))
    print(type(df.iloc[0]['image']['bytes']))

    return df


def generate_examples(data_dir):

        jsonl_path = os.path.join(data_dir, "data.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                record = json.loads(line)
                # print(record.keys())
                yield idx, {
                    "gt_parse": json.dumps(record["gt_parse"]),  # Store as string, or just `record["gt_parse"]` if you want dicts
                    "image": os.path.join(data_dir, record["image_path"])
                }


def inspect_data_examples(data_dir):
    dataset = list(generate_examples(data_dir))
    # View the first example
    example = dataset[0]
    print(dataset[0])

    print("Keys:", example[0])
    print("Keys:", example[1].keys())
    print("Parsed data:", example[1]["gt_parse"])
    # example[1]["image"].show()  # Uncomment if using PIL (will open the image)
 

def display_dataset_info(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Print basic info
    print(dataset)
    print(dataset['train'].features)

    # Access and display an example
    example = dataset["train"][0]
    # print(example)
    print("Parsed data:", json.loads(example["ground_truth"]))

    # Display the image
    img = example['image']
    img.show()


# Read local parquet files generated
read_and_inspect_parquet("./train.parquet")
read_and_inspect_parquet("./handwritten_archives_v3.parquet")


# Test and verify uploaded dataset
display_dataset_info("AqsaK/1880_census_handwritten_archives")   