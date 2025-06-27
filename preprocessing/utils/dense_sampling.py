import os
import random
import cv2
from glob import glob

def dense_sampling(video_path, num_clips=8, frames_per_clip=4):
    video_files = sorted(glob(os.path.join(video_path, '*.png'))) 
    num_frames = len(video_files)

    if num_frames < num_clips * frames_per_clip:
        raise ValueError("Not enough frames to sample the required clips.")

    frames_per_segment = num_frames // num_clips

    clips = []

    for i in range(num_clips):
        segment_start = i * frames_per_segment
        segment_end = segment_start + frames_per_segment - 1

        max_start_frame = segment_end - frames_per_clip + 1
        start_frame = random.randint(segment_start,  max_start_frame)
        
        clip = [cv2.imread(video_files[start_frame + j]) for j in range(frames_per_clip)]
        clips.append(clip)

    return clips