import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def remap_households_per_page(households):
    remapped_households = {
        str(new_id): households[old_key]
        for new_id, old_key in enumerate(households.keys(), start=1)
    }
    return remapped_households

def remap_flatten_households(households):
    # Flatten all people from all households
    all_people = []
    for members in households.values():
        all_people.extend(members)

    # Create a new structure for individuals
    # Replace structure
    remapped_households = {"individuals": all_people}
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

    #rename columns
    df = df.rename(columns={"FORNAMN": "FN", "ENAMN": "EN"})
    
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
            households = remap_flatten_households(households)
            json.dump({"ground_truth" : {"gt_parse" : {"households": households}}, "image_path" : image_path}, f, ensure_ascii=False)
            f.write("\n")

    return jsonl_path

def create_jsonl_splits(df, output_dir, train_frac=0.8, val_frac=0.1, seed=42):
    os.makedirs(output_dir, exist_ok=True)

    # Get unique pages (BILDNR)
    unique_pages = df["BILDNR"].unique()
    train_pages, temp_pages = train_test_split(unique_pages, train_size=train_frac, random_state=seed)
    val_pages, test_pages = train_test_split(temp_pages, test_size=0.5, random_state=seed)

    def subset_df(pages_subset):
        return df[df["BILDNR"].isin(pages_subset)]

    # Create and save splits
    convert_to_jsonl(subset_df(train_pages), os.path.join(output_dir, "train.jsonl"))
    convert_to_jsonl(subset_df(val_pages), os.path.join(output_dir, "val.jsonl"))
    convert_to_jsonl(subset_df(test_pages), os.path.join(output_dir, "test.jsonl"))



# def generate_parquet_file():
# # Load your original .jsonl data
#     records = []

#     with open("data.jsonl", "r", encoding="utf-8") as f:
#         for line in f:
#             record = json.loads(line)
#             image_path = record["image_path"]
#             print("image_path: ", image_path)
#             image_bytes = open(image_path, "rb").read()
#             gt = record["ground_truth"] # typo retained if intentional

#             records.append({
#                 "image": {"bytes": image_bytes},       # Store raw image
#                 "ground_truth": gt                     # Flatten: no more gt_parse key
#             })

#     df = pd.DataFrame(records)

#     # Save to parquet
#     df.to_parquet("./handwritten_archives_test.parquet", index=False)

# def generate_parquet_file():
# # Load your original .jsonl data
#     records = []
#     k=0

#     with open("data.jsonl", "r", encoding="utf-8") as f:
#         for line in f:
#             record = json.loads(line)
#             image_path = record["image_path"]
#             image_bytes = open(image_path, "rb").read()
#             gt = record["ground_truth"] # typo retained if intentional        

#             records.append({
#                 "image": {"bytes": image_bytes},       # Store raw image
#                 "ground_truth": json.dumps(gt)         
#             })
#             k+=1
#             if k % 100 == 0:
#                 print(f"Processed {k} records")

#     df = pd.DataFrame(records)

#     # Save to parquet
#     df.to_parquet("./train.parquet", index=False)


import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

def generate_parquet_file(batch_size=5000):
    records = []
    k = 0
    batch_id = 0

    parquet_writer = None

    with open("data.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            image_path = record["image_path"]
            
            try:
                with open(image_path, "rb") as img_f:
                    image_bytes = img_f.read()
            except FileNotFoundError:
                print(f"Missing image: {image_path}")
                continue
            
            gt = record["ground_truth"]

            records.append({
                "image": image_bytes,
                "ground_truth": json.dumps(gt)
            })

            k += 1

            if k % batch_size == 0:
                print(f"Processed {k} records")
                df = pd.DataFrame(records)

                # Convert to pyarrow Table
                table = pa.Table.from_pandas(df)

                # Write to Parquet incrementally
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter("train.parquet", table.schema)
                parquet_writer.write_table(table)

                records = []  # clear memory
                batch_id += 1

        # Write any remaining records
        if records:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter("train.parquet", table.schema)
            parquet_writer.write_table(table)

    if parquet_writer:
        parquet_writer.close()

    print(f"✅ Done. Total processed: {k} rows.")


def generate_jsonl_file():

    # cols = ['ID', 'RAD', 'FORNAMN', 'ENAMN', 'HNR','FNR','BILDNR', 'YRKE', 'KON', 'CIV', 'FODAR', 'FODORT', 'FODFORS', 'KYRKORT', 'LYTE', 'NATIONAL']
    cols = ['ID', 'RAD', 'FORNAMN', 'ENAMN', 'HNR', 'BILDNR']


    # Read the text file
    df = pd.read_csv("../data_preprocessing/original_data/1880_census_databasuttag.txt", sep="\t", dtype=str, encoding="latin1")
    df = df[cols]

    print("Dataframe read successfully.")

    # Use the function to filter the dataframe
    image_folder = 'images'
    filtered_df = filter_dataframe_by_images(df, image_folder)

    print("Dataframe filtered successfully.")

    # Convert the filtered dataframe and save as JSONL
    filtered_output_jsonl_path = "./data.jsonl"
    convert_to_jsonl(filtered_df, filtered_output_jsonl_path)

    print("Dataframe converted to JSONL successfully.")


# generate_jsonl_file()
# generate_parquet_file()