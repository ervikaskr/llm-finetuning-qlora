"""Download the Amazon Bestselling Books dataset from Kaggle."""
import shutil
from pathlib import Path

import kagglehub

DATA_DIR = Path(__file__).parent.parent / "data"

path = kagglehub.dataset_download("shambhurajejagadale/amazon-bestselling-books-dataset-500-books")
print(f"Downloaded to: {path}")

# Copy CSV to project data/ directory
DATA_DIR.mkdir(exist_ok=True)
for csv_file in Path(path).glob("*.csv"):
    dest = DATA_DIR / csv_file.name
    shutil.copy2(csv_file, dest)
    print(f"Copied → {dest}")
