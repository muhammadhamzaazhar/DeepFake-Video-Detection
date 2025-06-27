import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob, random
from torch.utils.data import Subset

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.video_dirs = []
        self.labels = []

        for label, subdir in enumerate(['real', 'fake']):
            subdir_path = os.path.join(root_dir, subdir)
            video_dirs = [
                os.path.join(subdir_path, video) 
                for video in os.listdir(subdir_path) 
                if os.path.isdir(os.path.join(subdir_path, video))
            ]
            self.video_dirs.extend(video_dirs)
            self.labels.extend([label] * len(video_dirs))

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, idx):
        video_dir = self.video_dirs[idx]
        label = self.labels[idx]

        frame_paths = sorted(glob.glob(os.path.join(video_dir, '*.png')))[:self.num_frames]
        if len(frame_paths) < self.num_frames:
            raise ValueError(f"Video {video_dir} has fewer than {self.num_frames} frames.")

        frames = []
        for frame_path in frame_paths:
            try:
                frame = Image.open(frame_path)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Error loading {frame_path}: {e}")
                raise

        frames_tensor = torch.stack(frames) 
        return {
            "pixel_values": frames_tensor,
            "labels": label
        }

# For timesformer
mean=[0.45, 0.45, 0.45]
std=[0.225, 0.225, 0.225]

# For swin_b
# mean=[0.485, 0.456, 0.406]
# std=[0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                   
    transforms.Normalize(               
        mean=mean,
        std=std
    )
])

def get_subset(dataset, percentage):
    total_size = len(dataset)
    subset_size = int(total_size * percentage)
    indices = list(range(total_size))
    random.shuffle(indices)
    selected_indices = indices[:subset_size]
    return Subset(dataset, selected_indices)

dataset_root = r'E:\Dataset_Thumbnail'
train_dataset = VideoFrameDataset(os.path.join(dataset_root, 'train'), num_frames=16, transform=transform)
val_dataset = VideoFrameDataset(os.path.join(dataset_root, 'val'), num_frames=16, transform=transform)
test_dataset = VideoFrameDataset(os.path.join(dataset_root, 'test'), num_frames=16, transform=transform)

random.seed(42)
train_dataset = get_subset(train_dataset, 0.3)
val_dataset = get_subset(val_dataset, 0.3)
test_dataset = get_subset(test_dataset, 0.3)

# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
