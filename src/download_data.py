import os
import shutil
import kagglehub

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at {DATA_DIR}")

    print("Downloading dataset from Kaggle via kagglehub...")
    # Download latest version of the dataset
    downloaded_path = kagglehub.dataset_download("xjoannax88/spacex-falcon-9-launches")
    print(f"Path to dataset files in cache: {downloaded_path}")
    
    # Copy all files from downloaded_path to DATA_DIR
    files = os.listdir(downloaded_path)
    for filename in files:
        src_file = os.path.join(downloaded_path, filename)
        dest_file = os.path.join(DATA_DIR, filename)
        if os.path.isdir(src_file):
            if os.path.exists(dest_file):
                shutil.rmtree(dest_file)
            shutil.copytree(src_file, dest_file)
            print(f"Copied folder {filename} to {DATA_DIR}")
        else:
            shutil.copy2(src_file, dest_file)
            print(f"Copied file {filename} to {DATA_DIR}")

if __name__ == '__main__':
    download_data()
