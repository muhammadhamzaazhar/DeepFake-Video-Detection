import sys, os, io, random, time, shutil, glob
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from contextlib import redirect_stdout
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from transformers import TimesformerForVideoClassification
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt
from colorama import init, Fore, Style

from preprocessing.utils.config import VideoConfig

init(autoreset=True)

FRAME_RATE = VideoConfig.FRAME_RATE
LOWCUT = VideoConfig.LOWCUT
HIGHCUT = VideoConfig.HIGHCUT
FRAME_SIZE = VideoConfig.FRAME_SIZE 
NUM_FRAMES_PER_CLIP = VideoConfig.NUM_FRAMES_PER_CLIP 
THUMBNAIL_SIZE = VideoConfig.THUMBNAIL_SIZE
GRID_ROWS = VideoConfig.GRID_ROWS
GRID_COLS = VideoConfig.GRID_COLS

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                   
    transforms.Normalize(               
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225]
    )
])

class VideoProcessor:
    def __init__(self, frame_size=(224, 224), num_frames=16):
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.detector = MTCNN()

    def detect_faces_in_video(self, video_path, padding_percentage=0.3, full_detection_interval=10):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"{Fore.RED}[ERROR] Unable to open video file {video_path}{Style.RESET_ALL}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        cropped_faces = []
        trackers = []

        with tqdm(total=total_frames, desc="Extracting faces", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame is None:
                    print(f"{Fore.YELLOW}[WARNING] Empty frame at {frame_count}{Style.RESET_ALL}")
                    continue

                if frame_count % full_detection_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = self.detector.detect_faces(rgb_frame)
                    trackers = []

                    for face in faces:
                        if face['confidence'] < 0.85:
                            continue

                        x, y, w, h = face['box']
                        if w < 50 or h < 50:
                            continue

                        padding = max(1, int(min(w, h) * padding_percentage))
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(frame.shape[1], x + w + padding)
                        y2 = min(frame.shape[0], y + h + padding)

                        cropped_face = frame[y1:y2, x1:x2]
                        if cropped_face.size == 0:
                            continue

                        resized_cropped_face = cv2.resize(cropped_face, self.frame_size)
                        cropped_faces.append(resized_cropped_face)

                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x, y, w, h))
                        trackers.append(tracker)

                else:
                    for tracker in trackers:
                        success, box = tracker.update(frame)
                        if success:
                            x, y, w, h = [int(v) for v in box]
                            padding = max(1, int(min(w, h) * padding_percentage))
                            x1 = max(0, x - padding)
                            y1 = max(0, y - padding)
                            x2 = min(frame.shape[1], x + w + padding)
                            y2 = min(frame.shape[0], y + h + padding)

                            cropped_face = frame[y1:y2, x1:x2]
                            if cropped_face.size == 0:
                                continue

                            resized_cropped_face = cv2.resize(cropped_face, self.frame_size)
                            cropped_faces.append(resized_cropped_face)

                frame_count += 1
                pbar.update(1)

        cap.release()

        return cropped_faces 
    
    def dense_sampling(self, cropped_faces, num_clips=8, frames_per_clip=4):
        num_frames = len(cropped_faces)
        if num_frames < num_clips * frames_per_clip:
            raise ValueError(f"{Fore.RED}[ERROR] Not enough cropped frames ({num_frames}) to sample {num_clips} clips with {frames_per_clip} frames each{Style.RESET_ALL}")

        frames_per_segment = num_frames // num_clips
        clips = []

        for i in range(num_clips):
            segment_start = i * frames_per_segment
            segment_end = segment_start + frames_per_segment - 1
            max_start_frame = segment_end - frames_per_clip + 1
            start_frame = random.randint(segment_start, max_start_frame)

            clip = [cropped_faces[start_frame + j] for j in range(frames_per_clip)]
            clips.append(clip)

        return clips
    
    def skin_segmentation(self, image):
        """
        Perform skin segmentation on an image using color thresholding in the YCrCb color space.
        Returns a binary mask of the skin region.
        """
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
        return skin_mask
     
    def apply_skin_mask_to_frame(self, image):
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_faces(rgb_frame)

        if result and len(result) > 0:
            face = result[0]
            box = face['box']  
            x, y, w, h = box
            
            img_width, img_height = image.shape[:2]
            left = max(0, int(x))
            top = max(0, int(y))
            right = min(img_width, int(x + w))
            bottom = min(img_height, int(y + h))
        
            face_region = rgb_frame[top:bottom, left:right]
            skin_mask = self.skin_segmentation(face_region)
            skin_region = cv2.bitwise_and(face_region, face_region, mask=skin_mask)
            resized_skin_region = cv2.resize(skin_region, self.frame_size)

            return resized_skin_region
        
    def bandpass_filter(self,signal, fs, low, high, order=4):
        nyquist = 0.5 * fs
        low_norm = low / nyquist
        high_norm = high / nyquist
        b, a = butter(order, [low_norm, high_norm], btype='band')
        return filtfilt(b, a, signal, axis=0)
    
    def generate_toi_heatmaps_from_frames(self, masked_frames, frame_rate=FRAME_RATE, lowcut=LOWCUT, highcut=HIGHCUT):
        if len(masked_frames) != 32:
            raise ValueError(f"{Fore.RED}[ERROR] Expected 32 frames, but got {len(masked_frames)}{Style.RESET_ALL}")

        bitplanes_all_frames = []

        for frame in masked_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            H, W, _ = frame_rgb.shape
            bitplanes = np.zeros((H, W, 24), dtype=np.uint8)

            for c in range(3): 
                channel = frame_rgb[:, :, c]
                for bit in range(8):
                    bitplanes[:, :, c * 8 + bit] = (channel >> bit) & 1

            bitplanes_all_frames.append(bitplanes)

        X = np.array([[np.mean(bp[:, :, j]) for j in range(24)] for bp in bitplanes_all_frames]) 
        X_filtered = self.bandpass_filter(X, frame_rate, lowcut, highcut)

        variances = np.var(X_filtered, axis=0)
        selected_cols = np.where(variances > 1e-6)[0]

        if len(selected_cols) == 0:
            raise RuntimeError(f"{Fore.RED}[ERROR] All ICA input columns have near-zero variance{Style.RESET_ALL}")

        X_selected = X_filtered[:, selected_cols]
        ica = FastICA(n_components=len(selected_cols), random_state=0, max_iter=500)
        S = ica.fit_transform(X_selected)

        best_comp = np.argmax(np.std(S, axis=0))
        mixing_coeffs = np.zeros(24)
        mixing_coeffs[selected_cols] = ica.mixing_[:, best_comp]

        heatmaps = []
        for bitplanes in bitplanes_all_frames:
            weighted = sum(bitplanes[:, :, j].astype(np.float32) * mixing_coeffs[j] for j in range(24))
            normed = cv2.normalize(weighted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            normed = normed.astype(np.uint8)
            heatmap = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
            heatmaps.append(heatmap)

        return heatmaps
    
    def generate_thumbnail(self, frames):
        """ Rearrange frames into grid layout, where each frame is resized to sub-image size (C × H/√t x W/√t)"""

        assert frames.shape[0] == NUM_FRAMES_PER_CLIP, "Mismatch in the number of frames per clip"

        sub_image_height = FRAME_SIZE // GRID_ROWS
        sub_image_width = FRAME_SIZE // GRID_COLS

        resized_frames = tf.image.resize(frames, (sub_image_height, sub_image_width))
        frames_grid = tf.reshape(
            resized_frames,
            (GRID_ROWS, GRID_COLS, sub_image_height, sub_image_width, frames.shape[-1])
        )  # Reshape to grid layout (r x c x H x W x C)

        frames_grid = tf.transpose(frames_grid, perm=[0, 2, 1, 3, 4])  # Rearrange grid to (r*H x c*W x C)
        frames_grid = tf.reshape(frames_grid, (GRID_ROWS * sub_image_height, GRID_COLS * sub_image_width, frames.shape[-1]))

        thumbnail = tf.image.resize(frames_grid, THUMBNAIL_SIZE)
        return thumbnail.numpy()
        
def save_thumbs(thumbs, prefix, out_dir):
    for idx, rgb_thumb in enumerate(thumbs, start=1):
        fn = f"{prefix}_{idx:02d}.png"
        if rgb_thumb.dtype != np.uint8:
            if rgb_thumb.max() <= 1.0:
                rgb_thumb = (rgb_thumb * 255).astype(np.uint8)
            else:
                rgb_thumb = rgb_thumb.astype(np.uint8)
        BGR = cv2.cvtColor(rgb_thumb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, fn), BGR)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"\n{Fore.RED}[ERROR] Usage: python predict.py <video_path>{Style.RESET_ALL}")
        sys.exit(1)
        
    video_path = sys.argv[1]
    if not os.path.exists(video_path) or not os.path.isfile(video_path):
        print(f"\n{Fore.RED}[ERROR] The provided video path does not exist or is not a file: {video_path}{Style.RESET_ALL}")
        sys.exit(1)

    with redirect_stdout(io.StringIO()):
        video = VideoFileClip(video_path)
    duration = int(video.duration)  
    video.reader.close()

    try:
        print(f"\n{Fore.CYAN}=== Starting Video Processing ==={Style.RESET_ALL}")
        video_processor = VideoProcessor()

        root_dir = "video_processing_outputs"
        if os.path.exists(root_dir):
            try:
                shutil.rmtree(root_dir)
                print(f"{Fore.GREEN}[INFO] Deleted existing directory: {root_dir}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}[ERROR] Failed to delete {root_dir}: {e}{Style.RESET_ALL}")
                sys.exit(1)

        sampled_frames_dir = os.path.join(root_dir, "sampled_frames")
        masked_frames_dir = os.path.join(root_dir, "masked_frames")
        toi_heatmaps_dir = os.path.join(root_dir, "toi_heatmaps")
        thumbnails_dir = os.path.join(root_dir, "all_thumbnails")

        print(f"\n{Fore.CYAN}=== Face Extraction ==={Style.RESET_ALL}")
        print(f"{Fore.BLUE}[INFO] Processing video: {video_path} (Duration: {duration} seconds){Style.RESET_ALL}")
        time_start = time.time()
        cropped_faces = video_processor.detect_faces_in_video(video_path)
        time_end = time.time()
        elapsed_time = time_end - time_start
        print(f"{Fore.GREEN}[SUCCESS] Extracted {len(cropped_faces)} face crops in {elapsed_time:.2f} seconds{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}=== Dense Sampling ==={Style.RESET_ALL}")
        print(f"{Fore.BLUE}[INFO] Sampling clips from {len(cropped_faces)} face crops{Style.RESET_ALL}")
        sampled_clips = video_processor.dense_sampling(cropped_faces, num_clips=8, frames_per_clip=4)
        print(f"{Fore.GREEN}[SUCCESS] Sampled {len(sampled_clips)} clips with {len(sampled_clips[0])} frames each{Style.RESET_ALL}")

        print(f"{Fore.BLUE}[INFO] Saving sampled frames to {sampled_frames_dir}{Style.RESET_ALL}")
        os.makedirs(sampled_frames_dir, exist_ok=True)
        for clip_idx, clip in enumerate(sampled_clips):
            for frame_idx, frame in enumerate(clip):
                face_filename = f"clip_{clip_idx + 1}_frame_{frame_idx + 1}.png"
                face_path = os.path.join(sampled_frames_dir, face_filename)
                cv2.imwrite(face_path, frame)      

        print(f"\n{Fore.CYAN}=== Skin Masking ==={Style.RESET_ALL}")
        print(f"{Fore.BLUE}[INFO] Applying skin mask to sampled frames{Style.RESET_ALL}")
        os.makedirs(masked_frames_dir, exist_ok=True)
        masked_frames = []
        for clip_idx in range(1, 9):
            for frame_idx in range(1, 5):
                frame_path = os.path.join(sampled_frames_dir, f"clip_{clip_idx}_frame_{frame_idx}.png")
                frame = cv2.imread(frame_path)

                if frame is None:
                    print(f"{Fore.RED}[ERROR] Unable to read frame {frame_path}{Style.RESET_ALL}")
                    continue

                masked_frame = video_processor.apply_skin_mask_to_frame(frame)

                if masked_frame is not None and masked_frame.shape[0] > 0 and masked_frame.shape[1] > 0:
                    masked_filename = f"clip_{clip_idx}_frame_{frame_idx}.png"
                    masked_path = os.path.join(masked_frames_dir, masked_filename)
                    cv2.imwrite(masked_path, cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR))
                    masked_frames.append(masked_frame)
                else:
                    print(f"{Fore.YELLOW}[WARNING] Skipping invalid/empty masked frame at clip {clip_idx}, frame {frame_idx}{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}[SUCCESS] Applied skin mask to {len(masked_frames)} frames{Style.RESET_ALL}")
        print(f"{Fore.BLUE}[INFO] Saved masked frames to {masked_frames_dir}{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}=== TOI Feature Extraction ==={Style.RESET_ALL}")
        print(f"{Fore.BLUE}[INFO] Generating TOI heatmaps from {len(masked_frames)} masked frames{Style.RESET_ALL}")
        os.makedirs(toi_heatmaps_dir, exist_ok=True)
        toi_heatmaps = video_processor.generate_toi_heatmaps_from_frames(masked_frames)
        for idx, heatmap in enumerate(toi_heatmaps):
            heatmap_filename = f"toi_heatmap_{idx+1}.png"
            heatmap_path = os.path.join(toi_heatmaps_dir, heatmap_filename)
            cv2.imwrite(heatmap_path, heatmap)
        print(f"{Fore.GREEN}[SUCCESS] Generated {len(toi_heatmaps)} TOI heatmaps{Style.RESET_ALL}")
        print(f"{Fore.BLUE}[INFO] Saved heatmaps to {toi_heatmaps_dir}{Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}=== Thumbnail Generation ==={Style.RESET_ALL}")
        print(f"{Fore.BLUE}[INFO] Generating optical and TOI thumbnails{Style.RESET_ALL}")
        optical_thumbnails = []
        toi_thumbnails = []
        for clip in sampled_clips:
            clip_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in clip]
            frames_np = np.stack(clip_rgb, axis=0)
            thumb = video_processor.generate_thumbnail(frames_np)
            optical_thumbnails.append(thumb)

        frames_per_clip = NUM_FRAMES_PER_CLIP
        n_toi_clips = len(toi_heatmaps) // frames_per_clip

        for i in range(n_toi_clips):
            start = i * frames_per_clip
            end = start + frames_per_clip
            heatmap_clip = toi_heatmaps[start:end]
            clip_rgb = [cv2.cvtColor(hm, cv2.COLOR_BGR2RGB) for hm in heatmap_clip]
            frames_np = np.stack(clip_rgb, axis=0)
            thumb = video_processor.generate_thumbnail(frames_np)
            toi_thumbnails.append(thumb)

        os.makedirs(thumbnails_dir, exist_ok=True)
        save_thumbs(optical_thumbnails, "opt_thumb", thumbnails_dir)
        save_thumbs(toi_thumbnails, "toi_thumb", thumbnails_dir)
        print(f"{Fore.GREEN}[SUCCESS] Saved {len(optical_thumbnails)} optical and {len(toi_thumbnails)} TOI thumbnails to {thumbnails_dir}{Style.RESET_ALL}")

        thumbnails_array = []
        for i, thumb in enumerate(optical_thumbnails + toi_thumbnails):
            if thumb is not None and thumb.size > 0:
                thumbnails_array.append(thumb)
            else:
                print(f"{Fore.YELLOW}[WARNING] Skipping invalid thumbnail{Style.RESET_ALL}")
        

        print(f"\n{Fore.CYAN}=== Inference Started ==={Style.RESET_ALL}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TimesformerForVideoClassification.from_pretrained("./weights/best_model").to(device)
        model.eval()

        # thumbnails_tensor = [
        #     torch.tensor(frame, dtype=torch.float32) if not isinstance(frame, torch.Tensor) else frame
        #     for frame in thumbnails_array
        # ]
        # transformed_frames = [transform(Image.fromarray(frame.astype('uint8'))) for frame in thumbnails_array]
        folder_path = r'./video_processing_outputs/all_thumbnails' 
        image_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")
        
        if len(image_files) != 16:
            raise ValueError(f"Expected 16 frames, but found {len(image_files)}")

        transformed_frames = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image: {img_path}")
    
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            transformed_img = transform(img)
            transformed_frames.append(transformed_img)

        video_tensor = torch.stack(transformed_frames)
        # video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = video_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(video_tensor)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        probabilities = probs.cpu().numpy()[0]
      
        real_prob = probabilities[0]
        fake_prob = probabilities[1]
    
        prediction = "REAL" if real_prob > fake_prob else "FAKE"
        confidence = max(real_prob, fake_prob)
   
        result_color = Fore.GREEN if prediction == "REAL" else Fore.RED
        confidence_color = Fore.YELLOW
  
        print(f"{result_color}Predicted: {prediction}{Style.RESET_ALL}")
        print(f"{confidence_color}Confidence: {confidence:.2%}{Style.RESET_ALL}")
        print(f"Detailed probabilities:")
        print(f"REAL: {real_prob:.2%}")
        print(f"FAKE: {fake_prob:.2%}")

    except Exception as e:
        print(f"{Fore.RED}[ERROR] An error occurred: {e}{Style.RESET_ALL}")
        sys.exit(1)

        