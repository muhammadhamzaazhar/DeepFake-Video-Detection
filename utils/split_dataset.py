import os
import shutil
import random
from pathlib import Path

dataset_path = "E:\Dataset_Thumbnail"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

random.seed(42)

output_dirs = ["train", "val", "test"]
for split in output_dirs:
    for category in ["real", "fake"]:
        Path(f"{dataset_path}/{split}/{category}").mkdir(parents=True, exist_ok=True)

def split_dataset():
    for category in ["real", "fake"]:
        category_path = os.path.join(dataset_path, category)
        video_files = os.listdir(category_path)

        random.shuffle(video_files)
        train_idx = int(len(video_files) * train_ratio)
        val_idx = int(len(video_files) * (train_ratio + val_ratio))

        train_files = video_files[:train_idx]
        val_files = video_files[train_idx:val_idx]
        test_files = video_files[val_idx:]

        for file in train_files:
            shutil.move(os.path.join(category_path, file), os.path.join(dataset_path, "train", category, file))
        for file in val_files:
            shutil.move(os.path.join(category_path, file), os.path.join(dataset_path, "val", category, file))
        for file in test_files:
            shutil.move(os.path.join(category_path, file), os.path.join(dataset_path, "test", category, file))

        os.rmdir(category_path)

split_dataset()
print("Dataset successfully split within the same directory!")