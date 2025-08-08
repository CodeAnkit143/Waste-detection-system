# src/split_dataset.py
import os, shutil
from sklearn.model_selection import train_test_split

# Set this to the path where you unzipped the Kaggle dataset
# Example: r"C:\Users\You\Downloads\garbage_classification" or "/home/you/garbage_classification"
DATASET_DIR = "archive\garbage_classification"
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..", "waste_split")

def split_data():
    # if DATASET_DIR == "archive\garbage_classification":
    #     raise ValueError("Please set DATASET_DIR at the top of this file to your dataset path.")
    for category in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, category)
        if not os.path.isdir(path):
            continue
        images = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
        train, test = train_test_split(images, test_size=0.2, random_state=42)
        train, val = train_test_split(train, test_size=0.1, random_state=42)

        for split, data in [('train', train), ('val', val), ('test', test)]:
            split_dir = os.path.join(OUTPUT_BASE, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for img in data:
                shutil.copy(os.path.join(path, img), os.path.join(split_dir, img))
    print("✅ Dataset split into train/val/test! Output base:", OUTPUT_BASE)

if __name__ == "__main__":
    split_data()
