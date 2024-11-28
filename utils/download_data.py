import os
import zipfile
import urllib.request

def download_and_extract_data():
    url = "URL_TO_YOUR_DATASET"  # Substitua pela URL do seu dataset
    data_dir = "data/"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    filename = url.split("/")[-1]
    filepath = os.path.join(data_dir, filename)

    urllib.request.urlretrieve(url, filepath)

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
