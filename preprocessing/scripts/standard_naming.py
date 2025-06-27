import os
import sys
from PIL import Image

INPUT_DIR = r"E:\Dataset\fake"  

def resize_image(image_path, size=(224, 224)):
    """
    Resize an image to the specified size using high-quality resampling.
    """
    try:
        with Image.open(image_path) as img:
            # Use LANCZOS resampling for high quality
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(image_path)  
        return True
    except Exception as e:
        print(f"Error resizing {image_path}: {str(e)}")
        return False

def process_directory(input_dir):
    os.chdir(input_dir)
    video_dirs = sorted([d for d in os.listdir() if os.path.isdir(d) and d.startswith("video")], 
                       key=lambda x: int(x[5:]))
    
    for video_dir in video_dirs:
        os.chdir(video_dir)
        print(f"Processing {video_dir}...")
        
        png_files = sorted([f for f in os.listdir() if f.endswith(".png")])
        
        if len(png_files) != 32:
            print(f"Warning: {video_dir} contains {len(png_files)} PNGs (expected 32), skipping")
            os.chdir("..")
            continue

        for i, png_file in enumerate(png_files, 1):
            clip_num = ((i-1) // 4) + 1  # 1-8
            frame_num = ((i-1) % 4) + 1   # 1-4
            
            new_name = f"clip_{clip_num}_frame_{frame_num}.png"  
            
            try:
                os.rename(png_file, new_name)
                print(f"Renamed {png_file} to {new_name}")
          
                resize_image(new_name)
            except Exception as e:
                print(f"Error renaming {png_file}: {str(e)}")
        
        os.chdir("..")  
    
    os.chdir("..") 

if __name__ == "__main__":
    if not os.path.isdir(INPUT_DIR):
        print(f"Error: The input directory '{INPUT_DIR}' does not exist.")
        sys.exit(1)
 
    process_directory(INPUT_DIR)