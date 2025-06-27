import os

path = r'G:\DeepFakeDataset\Dataset\real'

subfolders = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

temp_names = {}
for idx, folder in enumerate(subfolders, start=1):
    current_path = os.path.join(path, folder)
    temp_name = f"temp_video_{idx}"
    temp_path = os.path.join(path, temp_name)
    
    os.rename(current_path, temp_path)
    temp_names[temp_name] = folder  

counter = 1
for temp_name in temp_names.keys():
    temp_path = os.path.join(path, temp_name)
    new_folder_name = f"video{counter}"
    new_path = os.path.join(path, new_folder_name)

    os.rename(temp_path, new_path)
    print(f"Renamed '{temp_name}' to '{new_folder_name}'")
    
    counter += 1

print("Renaming completed!")