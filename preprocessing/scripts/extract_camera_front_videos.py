import os
import shutil

root_folder = r'G:\DeepFakeDataset\source_videos_part_06' 
destination_folder = r'G:\DeepFakeDataset\dataset_real_videos'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

for current_root, dirs, files in os.walk(root_folder):
    if os.path.basename(current_root) == 'camera_front':
        print(f"Found camera_front folder: {current_root}")
        for file in files:
            source_file = os.path.join(current_root, file)
            if not file.lower().endswith(('.mp4')):
                continue
            
            destination_file = os.path.join(destination_folder, file)

            shutil.copy2(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")

print("Processing completed!")
