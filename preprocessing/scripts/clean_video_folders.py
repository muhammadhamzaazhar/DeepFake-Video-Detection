import os
import shutil

path = r'E:\Dataset_Thumbnail\real'

deleted_count = 0

for video_folder in os.listdir(path):
    video_path = os.path.join(path, video_folder)
    
    if os.path.isdir(video_path):
        frames = os.listdir(video_path)
        
        if len(frames) != 16:
            print(f"Deleting folder: {video_folder} (contains {len(frames)} files)")
            shutil.rmtree(video_path)
            deleted_count += 1

print(f"Total deleted video folders: {deleted_count}")
