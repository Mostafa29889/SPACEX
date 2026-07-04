import os
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

DATASETS = {
    'dataset_part_1.csv': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_1.csv',
    'dataset_part_2.csv': 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv'
}

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")

    for filename, url in DATASETS.items():
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.exists(filepath):
            print(f"{filename} already exists at {filepath}. Skipping download.")
            continue
        
        print(f"Downloading {filename} from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Successfully saved to {filepath}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == '__main__':
    download_data()
