import os
import random
import string

folder_path = r'G:\DeepFakeDataset\dataset_real_videos'  

def generate_random_name(length=6):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    
    if os.path.isfile(file_path) and filename.lower().endswith('.mp4'):
        _, file_extension = os.path.splitext(filename)
        
        new_name = generate_random_name() + file_extension
        new_file_path = os.path.join(folder_path, new_name)
        
        while os.path.exists(new_file_path):
            new_name = generate_random_name() + file_extension
            new_file_path = os.path.join(folder_path, new_name)
        
        os.rename(file_path, new_file_path)
        print(f"Renamed '{filename}' to '{new_name}'")

print("Renaming completed!")
