"""
download_data.py
----------------
Script to ensure the IBM Telco Customer Churn dataset is available locally.
If the dataset is already present in the data/ directory, it skips downloading.
"""

import os
import urllib.request

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(DATA_DIR, "Telco-Customer-Churn.csv")
DATASET_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def ensure_dataset():
    if os.path.exists(FILE_PATH):
        file_size = os.path.getsize(FILE_PATH)
        print(f"[OK] Dataset already exists at: {FILE_PATH} ({file_size:,} bytes)")
        return FILE_PATH

    print(f"Dataset not found locally. Downloading from:\n{DATASET_URL} ...")
    try:
        urllib.request.urlretrieve(DATASET_URL, FILE_PATH)
        file_size = os.path.getsize(FILE_PATH)
        print(f"[SUCCESS] Download completed: {FILE_PATH} ({file_size:,} bytes)")
        return FILE_PATH
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        raise e

if __name__ == "__main__":
    ensure_dataset()
