import json
import pandas as pd
import os


def remap_households_per_page(households):
    remapped_households = {
        str(new_id): households[old_key]
        for new_id, old_key in enumerate(households.keys(), start=1)
    }
    return remapped_households

def filter_dataframe_by_images(df, image_folder):
    # Get list of image filenames
    image_filenames = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Extract BILDNR from image filenames (assuming the filenames are the BILDNR with an extension)
    image_bildnr = [os.path.splitext(f)[0] for f in image_filenames]

    # Check if 'BILDNR' column exists in the dataframe
    if 'BILDNR' not in df.columns:
        raise KeyError("The dataframe does not contain the 'BILDNR' column.")
    
    # Filter the dataframe based on BILDNR
    filtered_df = df[df['BILDNR'].isin(image_bildnr)]
    
    return filtered_df


def convert_to_jsonl(df, output_path):
    """
    Converts a dataframe into a JSONL format where each line represents a page (BILDNR)
    with households (HNR) containing individuals and their attributes.
    """
    pages = {}
    
    for _, row in df.iterrows():
        page_id = row["BILDNR"]
        household_id = row["HNR"]
        
        # Remove unnecessary columns for people attributes
        person_data = row.drop(["BILDNR", "HNR", "ID", "RAD"]).dropna().to_dict()

        if page_id not in pages:
            pages[page_id] = {}

        if household_id not in pages[page_id]:
            pages[page_id][household_id] = []

        pages[page_id][household_id].append(person_data)

    # Write to JSONL
    jsonl_path = output_path
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for page_id, households in pages.items():
            image_path = f"images/{page_id}.jpg"
            households = remap_households_per_page(households)
            json.dump({"ground_truth" : {"gt_parse" : {"households": households}}, "image_path" : image_path}, f, ensure_ascii=False)
            f.write("\n")

    return jsonl_path



def generate_parquet_file():
# Load your original .jsonl data
    records = []

    with open("../data/data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            image_path = record["image_path"]
            print("image_path: ", image_path)
            image_bytes = open(image_path, "rb").read()
            gt = record["ground_truth"] # typo retained if intentional

            records.append({
                "image": {"bytes": image_bytes},       # Store raw image
                "ground_truth": gt                     # Flatten: no more gt_parse key
            })

    df = pd.DataFrame(records)

    # Save to parquet
    df.to_parquet("./handwritten_archives_test_version.parquet", index=False)

def generate_jsonl_file():

    # Read the text file
    df = pd.read_csv("original_data/1880_census_databasuttag.txt", sep="\t", dtype=str, encoding="latin1")
    df = df[['ID', 'RAD', 'FORNAMN', 'ENAMN', 'HNR','FNR','BILDNR', 'YRKE', 'KON', 'CIV', 'FODAR', 'FODORT', 'FODFORS', 'KYRKORT', 'LYTE', 'NATIONAL']]

    print("Dataframe read successfully.")

    # Use the function to filter the dataframe
    image_folder = 'images'
    filtered_df = filter_dataframe_by_images(df, image_folder)

    print("Dataframe filtered successfully.")

    # Convert the filtered dataframe and save as JSONL
    filtered_output_jsonl_path = "./output_data/filtered_output_remapped.jsonl"
    convert_to_jsonl(filtered_df, filtered_output_jsonl_path)

    print("Dataframe converted to JSONL successfully.")


# generate_jsonl_file()
generate_parquet_file()