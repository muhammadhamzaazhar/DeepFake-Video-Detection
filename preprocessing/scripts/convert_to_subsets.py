import os
import re
import shutil

real_path = r'G:\DeepFakeDataset\dataset_real_videos'  
subset_size = 100 

video_files = sorted([
    f for f in os.listdir(real_path)
    if os.path.isfile(os.path.join(real_path, f)) and f.lower().endswith('.mp4')
])

total_videos = len(video_files)
num_new_subsets = (total_videos + subset_size - 1) // subset_size 

print(f"Total new video files: {total_videos}")
print(f"Creating {num_new_subsets} new subset folders each with up to {subset_size} videos.")

existing_subset_numbers = []
for item in os.listdir(real_path):
    item_path = os.path.join(real_path, item)
    if os.path.isdir(item_path):
        match = re.match(r'subset_(\d+)', item)
        if match:
            existing_subset_numbers.append(int(match.group(1)))

start_subset_number = max(existing_subset_numbers) + 1 if existing_subset_numbers else 1

print(f"Starting new subsets from: subset_{start_subset_number}")

for i in range(num_new_subsets):
    subset_folder = os.path.join(real_path, f'subset_{start_subset_number + i}')
    
    if not os.path.exists(subset_folder):
        os.mkdir(subset_folder)
 
    start = i * subset_size
    end = start + subset_size
    subset_videos = video_files[start:end]
    
    for video in subset_videos:
        src = os.path.join(real_path, video)
        dst = os.path.join(subset_folder, video)
        shutil.move(src, dst)
        print(f"Moved '{video}' to '{subset_folder}'")

print("Restructuring into subsets completed!")