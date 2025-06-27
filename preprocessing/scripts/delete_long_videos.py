import os
import io
from moviepy.video.io.VideoFileClip import VideoFileClip
from contextlib import redirect_stdout

directory = r'G:\DeepFakeDataset\dataset_real_videos\subset_24' 
video_extension = ('.mp4')
deleted_count = 0

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    
    if os.path.isfile(file_path) and filename.lower().endswith(video_extension):
        try:
            with redirect_stdout(io.StringIO()):
                clip = VideoFileClip(file_path)
            duration = clip.duration  
            clip.reader.close()  
            
            if duration > 60:
                os.remove(file_path)
                print(f"Deleted: {filename} (Duration: {duration:.2f} sec)")
                deleted_count += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print(f"Total deleted videos: {deleted_count}")