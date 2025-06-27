import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
import tempfile
import cv2
import numpy as np
import time

from utils.config import VideoConfig
from utils.frame_extraction import detect_faces_in_video
from utils.dense_sampling import dense_sampling
from utils.skin_masking import remove_non_skin_region
from utils.toi import generate_toi_heatmaps
from utils.thumbnail_generation import generate_thumbnail
from utils.thumbnail_generation import apply_mask

MASK_SIZE = VideoConfig.MASK_SIZE  # Size of the square mask (s x s)
FRAME_SIZE = VideoConfig.FRAME_SIZE
NUM_FRAMES_PER_CLIP = VideoConfig.NUM_FRAMES_PER_CLIP 
NUM_CLIPS = VideoConfig.NUM_CLIPS 
THUMBNAIL_SIZE = VideoConfig.THUMBNAIL_SIZE
GRID_ROWS = int(np.ceil(np.sqrt(NUM_FRAMES_PER_CLIP)))
GRID_COLS = int(np.ceil(NUM_FRAMES_PER_CLIP / GRID_ROWS))

def process_videos_for_face_detection(input_dir, output_dir, num_clips=8, frames_per_clip=4):
    print(f"Processing {input_dir} for face detection...")
    all_cropped_faces = {}
    valid_video_count = 1900  

    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return all_cropped_faces

    for video_filename in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_filename)
        if video_path.endswith('.mp4'):
            print(f"\nProcessing video: {video_filename}")

            with tempfile.TemporaryDirectory() as temp_dir:
                time_start = time.time()
                cropped_faces = detect_faces_in_video(video_path, temp_dir) 
                time_end = time.time()
                elapsed_time = time_end - time_start
                print(f"Number of cropped faces for {video_filename}: {len(cropped_faces)}. Total time taken: {elapsed_time:.2f} seconds")

                if cropped_faces:
                    valid_video_count += 1  
                    video_output_dir = os.path.join(output_dir, f"video{valid_video_count}")
                    os.makedirs(video_output_dir, exist_ok=True)

                    try:
                        print(f"Sampling frames from extracted frames.")
                        sampled_clips = dense_sampling(temp_dir, num_clips, frames_per_clip)
                        print(f"Total # of clips: {len(sampled_clips)}. Each clip contains 4 frames.")

                        sampled_face_paths = []
                        for clip_idx, clip in enumerate(sampled_clips):
                            for frame_idx, frame in enumerate(clip):
                                face_filename = f"clip_{clip_idx + 1}_frame_{frame_idx + 1}.png"
                                new_face_path = os.path.join(video_output_dir, face_filename)

                                cv2.imwrite(new_face_path, frame)
                                sampled_face_paths.append(new_face_path)

                        all_cropped_faces[video_filename] = sampled_face_paths        
                        print("Successfully sampled 32 frames from video.")
                    except ValueError as e:
                        print(f"Skipping {video_filename} due to sampling error: {e}")

                else:
                    print(f"Skipping {video_filename} due to face detection criteria.")

    return all_cropped_faces


def process_dataset_for_masked_frames(input_root, output_root):
    print(f"Processing {input_root} for masked frames...")

    for category in ['real', 'fake']:
        category_input_path = os.path.join(input_root, category)
        category_output_path = os.path.join(output_root, category)
        
        if not os.path.exists(category_input_path):
            print(f"Skipping {category_input_path} - directory not found")
            continue
        
        for video_name in os.listdir(category_input_path):
            video_input_path = os.path.join(category_input_path, video_name)
            video_output_path = os.path.join(category_output_path, video_name)
            
            if not os.path.isdir(video_input_path):
                continue
            
            os.makedirs(video_output_path, exist_ok=True)
            
            processed_count = 0
            print(f"Processing {video_input_path}")
            
            for frame_name in os.listdir(video_input_path):
                if frame_name.endswith('.png'): 
                    frame_input_path = os.path.join(video_input_path, frame_name)
                    frame_output_path = os.path.join(video_output_path, frame_name)
                    
                    if remove_non_skin_region(frame_input_path, frame_output_path):
                        processed_count += 1
            
            print(f"Processed {processed_count}/32 frames in {video_input_path}")
            if processed_count != 32:
                print(f"Warning: Expected 32 frames, but processed {processed_count} in {video_input_path}")


def process_masked_frames_for_toi_heatmaps(input_dir, output_dir):
    print(f"Processing {input_dir} for TOI heatmaps...")
    
    video_folders = [d for d in os.listdir(input_dir)
                     if os.path.isdir(os.path.join(input_dir, d))]
    
    for video_folder in video_folders:
        input_video_folder_path = os.path.join(input_dir, video_folder)
        output_video_folder_path = os.path.join(output_dir, video_folder)
        
        frame_files = sorted([f for f in os.listdir(input_video_folder_path)
                              if f.endswith('.png')])
        
        if len(frame_files) != 32:
            print(f"Skipping folder '{video_folder}' (found {len(frame_files)} frames, expected 32).")
            continue
        
        generate_toi_heatmaps(frame_files, input_video_folder_path, output_video_folder_path)


def process_samples_to_generate_thumbnails(folder_path, output_path):
    print(f"Processing {folder_path} for thumbnails...")
    
    os.makedirs(output_path, exist_ok=True)

    sub_h, sub_w = FRAME_SIZE // GRID_ROWS, FRAME_SIZE // GRID_COLS  # Sub-image size based on sqrt(T)

    for video_folder in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video_folder)

        if not os.path.isdir(video_path):
            print(f"Video path not found: {video_path}")
            continue

        video_output_path = os.path.join(output_path, video_folder)
        os.makedirs(video_output_path, exist_ok=True)

        for clip_index in range(1, NUM_CLIPS + 1):
            optical_frames = []
            toi_frames = []
            for frame_index in range(1, NUM_FRAMES_PER_CLIP + 1):
                optical_frame_path = os.path.join(video_path, f"clip_{clip_index}_frame_{frame_index}.png")
                toi_frame_path = os.path.join(video_path, f"clip_{clip_index}_heatmap_{frame_index}.png")

                if os.path.exists(optical_frame_path):
                    frame = cv2.imread(optical_frame_path)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        optical_frames.append(frame)
                    else:
                        print(f"Error reading optical frame: {optical_frame_path}")

                if os.path.exists(toi_frame_path):
                    frame = cv2.imread(toi_frame_path)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        toi_frames.append(frame)
                    else:
                        print(f"Error reading TOI frame: {toi_frame_path}")

            while len(optical_frames) < NUM_FRAMES_PER_CLIP:
                empty_frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
                optical_frames.append(empty_frame)

            while len(toi_frames) < NUM_FRAMES_PER_CLIP:
                empty_frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
                toi_frames.append(empty_frame)

            optical_frames = np.stack(optical_frames)  
            toi_frames = np.stack(toi_frames)         

            masked_optical = apply_mask(optical_frames, MASK_SIZE, sub_h, sub_w) 
            thumbnail_optical = generate_thumbnail(masked_optical)
            thumbnail_optical_bgr = cv2.cvtColor(thumbnail_optical, cv2.COLOR_RGB2BGR)
            save_path_optical = os.path.join(video_output_path, f"thumbnail_{clip_index}.png")
            cv2.imwrite(save_path_optical, thumbnail_optical_bgr)
            print(f"Thumbnail (Optical) saved for {video_folder} clip {clip_index} at {save_path_optical}")

            thumbnail_toi = generate_thumbnail(toi_frames)
            thumbnail_toi_bgr = cv2.cvtColor(thumbnail_toi, cv2.COLOR_RGB2BGR)
            save_path_toi = os.path.join(video_output_path, f"thumbnail_{clip_index}_toi.png")
            cv2.imwrite(save_path_toi, thumbnail_toi_bgr)
            print(f"Thumbnail (TOI) saved for {video_folder} clip {clip_index} at {save_path_toi}")


if __name__ == "__main__":
    INPUT_DIR = r"G:\DeepFakeDataset\dataset_real_videos\subset_12"
    OUTPUT_DIR = r"G:\DeepFakeDataset\Dataset\real"
    INPUT_DIR_FOR_MASKED_FRAMES = r"E:\Dataset"
    MASKED_OUTPUT_DIR = r"E:\Dataset_Masked"
    INPUT_MASKED_DIR = r"E:\Dataset_Masked\fake"
    OUTPUT_TOI_HEATMAP_DIR = r"E:\Dataset\fake"
    INPUT_SAMPLE_DIR = r"E:\Dataset\fake"
    OUTPUT_THUMBNAIL_DIR = r"E:\Dataset_Thumbnail\fake"

    print("Options:")
    print("  1 - Process videos for face detection")
    print("  2 - Process dataset for masked frames")
    print("  3 - Process masked frames for TOI heatmaps")
    print("  4 - Process samples to generate thumbnails")

    choice = input("Enter your choice (1, 2, 3, or 4): ")

    if choice == "1":
        process_videos_for_face_detection(INPUT_DIR, OUTPUT_DIR)
    elif choice == "2":
        process_dataset_for_masked_frames(INPUT_DIR_FOR_MASKED_FRAMES, MASKED_OUTPUT_DIR)
    elif choice == "3":
        process_masked_frames_for_toi_heatmaps(INPUT_MASKED_DIR, OUTPUT_TOI_HEATMAP_DIR)
    elif choice == "4":
        process_samples_to_generate_thumbnails(INPUT_SAMPLE_DIR, OUTPUT_THUMBNAIL_DIR)
    else:
        print(f"Invalid command: {choice}. Please enter 1, 2, 3, or 4.")
        sys.exit(1)