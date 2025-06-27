import os
import shutil

def remove_short_videos(dataset_path, min_frame_count=32):
    deleted_count = 0
    kept_count = 0

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(folder_path):
            continue  

        frame_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        num_frames = len(frame_files)

        if num_frames < min_frame_count:
            shutil.rmtree(folder_path)
            print(f"Deleted '{folder_name}' — only {num_frames} frames found.")
            deleted_count += 1
        else:
            kept_count += 1

    print(f"\nCompleted: {deleted_count} folders deleted. {kept_count} folders kept.")

dataset_root = "path/to/your/dataset"  
remove_short_videos(dataset_root)
