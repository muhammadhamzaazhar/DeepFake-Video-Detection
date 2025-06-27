import os
import shutil

def pad_video_folders_to_32(dataset_path):
    padded_count = 0

    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)

        if not os.path.isdir(folder_path):
            continue  

        frame_files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        )

        num_frames = len(frame_files)

        if num_frames == 28:
            last_frames = frame_files[-4:]
            for i, frame in enumerate(last_frames):
                new_frame = f"{num_frames + i:05d}.{frame.split('.')[-1]}"
                shutil.copy(
                    os.path.join(folder_path, frame),
                    os.path.join(folder_path, new_frame)
                )
            padded_count += 1
            print(f"Padded '{folder_name}' from 28 to 32 frames.")

        elif num_frames == 29:
            last_frames = frame_files[-3:]
            for i, frame in enumerate(last_frames):
                new_frame = f"{num_frames + i:05d}.{frame.split('.')[-1]}"
                shutil.copy(
                    os.path.join(folder_path, frame),
                    os.path.join(folder_path, new_frame)
                )
            padded_count += 1
            print(f"Padded '{folder_name}' from 29 to 32 frames.")

        elif num_frames == 30:
            last_frame_1 = frame_files[-2]
            last_frame_2 = frame_files[-1]

            new_frame_1 = f"{num_frames:05d}.{last_frame_1.split('.')[-1]}"
            new_frame_2 = f"{num_frames+1:05d}.{last_frame_2.split('.')[-1]}"

            shutil.copy(
                os.path.join(folder_path, last_frame_1),
                os.path.join(folder_path, new_frame_1)
            )
            shutil.copy(
                os.path.join(folder_path, last_frame_2),
                os.path.join(folder_path, new_frame_2)
            )
            padded_count += 1
            print(f"Padded '{folder_name}' from 30 to 32 frames.")

        elif num_frames == 31:
            last_frame = frame_files[-1]
            new_frame = f"{num_frames:05d}.{last_frame.split('.')[-1]}"

            shutil.copy(
                os.path.join(folder_path, last_frame),
                os.path.join(folder_path, new_frame)
            )
            padded_count += 1
            print(f"Padded '{folder_name}' from 31 to 32 frames.")

    print(f"\nDone. Padded {padded_count} folders to 32 frames.")

dataset_root = r"D:\DeepFake Datasets\DF40\test\mobileswap\mobileswap\cdf\frames\Youtu_Pangu_Security\public\youtu-pangu-public\zhiyuanyan\deepfakes_detection_datasets\DF40\mobileswap\cdf\frames"
pad_video_folders_to_32(dataset_root)
