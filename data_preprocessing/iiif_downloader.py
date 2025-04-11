# This script is designed to facilitate the downloading of images and manifests from the IIIF API 
# provided by the Swedish National Archives (Riksarkivet). It processes a dataset containing image 
# identifiers (BILDNR) and performs the following tasks:
# 1. Downloads images in JPG format using the IIIF API and saves them to a specified folder.
# 2. Fetches and saves the corresponding IIIF manifests in JSON format for each image.
# The script reads a tab-separated text file containing the image identifiers and processes each entry.
# Functions:
# - `download_images_as_jpg`: Downloads images from the IIIF API based on the image identifiers.
# - `download_manifest`: Downloads IIIF manifests for the corresponding image identifiers.
# Dependencies:
# - pandas: For reading and processing the input dataset.
# - requests: For handling HTTP requests.
# - json: For parsing and saving JSON data.
# - os: For file and directory operations.
# - urllib: For downloading files from URLs.
# Usage:
# - Ensure the input text file is available and contains a column with image identifiers (e.g., "BILDNR").
# - Update the file path and column name in the script as needed.
# - Run the script to download images and manifests to their respective folders.


import os
import requests
import pandas as pd
import json
import urllib.request


def download_images_as_jpg(df, image_column, save_folder="images"):
    """
    Downloads images from the IIIF API as JPG format based on the 'BILDNR' column.
    
    Args:
    df (pd.DataFrame): DataFrame containing the image IDs.
    image_column (str): Column name that holds the image numbers.
    save_folder (str): Directory to save downloaded images.
    """

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    df.dropna(subset=["BILDNR"], inplace=True)

    unique_bildnr = df['BILDNR'].unique()
    print(unique_bildnr)
    print(len(unique_bildnr))

    k= 0

    for image_id in unique_bildnr:
        # image_id = row[image_column]  # Get the BILDNR value

        # Construct the IIIF Manifest URL
        image_url = f"https://lbiiif.riksarkivet.se/folk!{image_id}/full/max/0/default.jpg"

        # print(image_url)

        # Define the file path to save the image
        image_path = os.path.join(save_folder, f"{image_id}.jpg")
        try:
            # Download the JPG image
            with urllib.request.urlopen(image_url, timeout=10) as img_response:
                with open(image_path, "wb") as file:
                    file.write(img_response.read())  # Save the image

            # print(f"Downloaded: {image_id}.jpg")
        except urllib.error.URLError as e:
            print(f"Failed to download {image_id}: {e}")
        except (KeyError, IndexError):
            print(f"Invalid manifest format for {image_id}")
        except Exception as e:
            print(f"An error occurred for {image_id}: {e}")
        k+=1
        if k % 100 == 0:
            print(f"Downloaded {k} images")
        

def download_manifest(df, image_column, save_folder="manifest"):
    """
    Downloads manifest based on the 'BILDNR' column.
    
    Args:
    df (pd.DataFrame): DataFrame containing the image IDs.
    image_column (str): Column name that holds the image numbers.
    save_folder (str): Directory to save downloaded manifest.
    """

    # Ensure the save folder exists
    os.makedirs(save_folder, exist_ok=True)

    import urllib.request

    for index, row in df[:2].iterrows():
        image_id = row[image_column]  # Get the BILDNR value

        # Construct the IIIF Manifest URL
        manifest_url = f"https://lbiiif.riksarkivet.se/folk!{image_id}/manifest"

        print(manifest_url)
        
        # try:
        # Fetch the manifest JSON
        with urllib.request.urlopen(manifest_url, timeout=10) as response:
            manifest_data = json.loads(response.read().decode())

        print("manifest_data: ", manifest_data)

        # Define the file path to save the manifest
        manifest_path = os.path.join(save_folder, f"{image_id}.json")

        # Save the manifest JSON to a file
        with open(manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest_data, file, ensure_ascii=False, indent=4)

        print(f"Saved manifest: {image_id}.json")


# Read the text file
df = pd.read_csv("original_data/1880_census_databasuttag.txt", sep="\t", dtype=str, encoding="latin1")

# Download Images
download_images_as_jpg(df, "BILDNR")

# Download the manifest
# download_manifest(df, "BILDNR")
