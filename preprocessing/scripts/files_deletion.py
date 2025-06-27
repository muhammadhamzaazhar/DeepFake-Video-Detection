import os

ROOT_DIR = r"E:\Dataset\real"

def delete_heatmap_files(root_dir):
    video_folders = [d for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))]

    for video_folder in video_folders:
        video_folder_path = os.path.join(root_dir, video_folder)
        print(f"Processing folder: {video_folder}")

        heatmap_files_to_delete = []
        for clip_num in range(1, 9): 
            for heatmap_num in range(1, 5): 
                filename = f"clip_{clip_num}_heatmap_{heatmap_num}.png"
                heatmap_files_to_delete.append(filename)

        deleted_count = 0
        for heatmap_file in heatmap_files_to_delete:
            file_path = os.path.join(video_folder_path, heatmap_file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    deleted_count += 1
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
            else:
                print(f"File does not exist: {file_path}")

        print(f"Finished processing '{video_folder}': {deleted_count} files deleted.")

if __name__ == "__main__":
    print(f"Starting deletion process in {ROOT_DIR}")
    delete_heatmap_files(ROOT_DIR)
    print("Deletion process completed.")