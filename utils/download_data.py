import os
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

def download_and_extract_data(dataset_name, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    api = KaggleApi()
    api.authenticate()

    dataset_path = f"{dataset_name}.zip"
    api.dataset_download_files(dataset_name, path=target_dir, unzip=True)

    print(f"Dados extraídos para {target_dir}")

dataset_name = 'cats-vs-dogs'
target_dir = 'data/'

download_and_extract_data(dataset_name, target_dir)
