import os
import logging
import hashlib
from pathlib import Path
import pandas as pd
import kaggle


logging.basicConfig(level=logging.INFO)
data_logger = logging.getLogger(__name__)

DATA_DIR = "/data/raw/"

def download_retailrocket():
    dataset = "retailrocket/ecommerce-dataset"
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    kaggle.api.dataset_download_files(dataset, path=str(data_dir), unzip=True)
    data_logger.info("Data has been downloaded into %s", data_dir)
    
    # Check hash for traceability
    for file in data_dir.glob("*.csv"):
        with open(file, "rb") as f:
            hash_md5 = hashlib.md5(f.read()).hexdigest()
        data_logger.info("%s: MD5 %s", file.name, hash_md5)

def get_events() -> pd.DataFrame:
    return pd.read_csv("data/raw/events.csv")

def get_item_properties() -> pd.DataFrame:
    item_props1 = pd.read_csv("data/raw/item_properties_part1.csv")
    item_props2 = pd.read_csv("data/raw/item_properties_part2.csv")
    item_props = pd.concat([item_props1, item_props2])
    return item_props

def get_categories() -> pd.DataFrame:
    return pd.read_csv("data/raw/category_tree.csv")
