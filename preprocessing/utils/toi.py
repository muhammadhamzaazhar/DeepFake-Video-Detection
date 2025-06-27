import os
import cv2
import numpy as np
from sklearn.decomposition import FastICA
from scipy.signal import butter, filtfilt

FRAME_RATE = 30  
LOWCUT = 0.7     # ~42 BPM
HIGHCUT = 2.0    # ~120 BPM

def bandpass_filter(signal, fs, low, high, order=4):
    nyquist = 0.5 * fs
    low_norm = low / nyquist
    high_norm = high / nyquist
    b, a = butter(order, [low_norm, high_norm], btype='band')
    filtered = filtfilt(b, a, signal, axis=0)
    return filtered

def generate_toi_heatmaps(frame_files, video_folder_path, output_folder, frame_rate=FRAME_RATE, lowcut=LOWCUT, highcut=HIGHCUT):
    """
    Generate TOI heatmaps from masked frames using ICA.
    
    Args:
        frame_files (list): List of frame filenames.
        video_folder_path (str): Path to the folder containing frames.
        output_folder (str): Path to save the heatmaps.
        frame_rate (int): Frame rate of the video.
        lowcut (float): Low frequency cutoff for bandpass filter (Hz).
        highcut (float): High frequency cutoff for bandpass filter (Hz).
    """
    bitplanes_all_frames = []  # (H, W, 24) per frame

    for frame_file in frame_files:
        frame_path = os.path.join(video_folder_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read {frame_path}, skipping.")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W, _ = frame_rgb.shape

        bitplanes = np.zeros((H, W, 24), dtype=np.uint8)
        for c in range(3):  
            channel = frame_rgb[:, :, c]
            for bit in range(8):
                bitplanes[:, :, c*8 + bit] = ((channel >> bit) & 1)

        bitplanes_all_frames.append(bitplanes)

    if len(bitplanes_all_frames) != 32:
        print(f"Expected 32 frames, but got {len(bitplanes_all_frames)}. Skipping.")
        return

    # Build Temporal Signals
    X = np.array([[np.mean(bitplanes[:, :, j]) for j in range(24)] for bitplanes in bitplanes_all_frames])  # Shape: (32, 24)

    X_filtered = bandpass_filter(X, frame_rate, lowcut, highcut, order=4)

    # Check variance of each column to exclude constant signals
    variance = np.var(X_filtered, axis=0)
    threshold = 1e-6  # Small threshold to detect near-constant columns
    selected_cols = np.where(variance > threshold)[0]
    
    if len(selected_cols) == 0:
        print("All columns have zero variance. Skipping.")
        return

    print(f"Selected {len(selected_cols)} out of 24 columns for ICA.")

    # Apply ICA only on columns with sufficient variance
    X_selected = X_filtered[:, selected_cols]
    ica = FastICA(n_components=len(selected_cols), random_state=0, max_iter=500)
    S = ica.fit_transform(X_selected)  # Shape: (32, n_selected)

    # Choose the component with the highest standard deviation
    comp_std = np.std(S, axis=0)
    selected_comp_idx = np.argmax(comp_std)
    print(f"[{os.path.basename(video_folder_path)}] Selected ICA component: {selected_comp_idx}")

    # Mixing coefficients for the selected component
    mixing_coeffs_selected = ica.mixing_[:, selected_comp_idx]  # Shape: (n_selected,)

    # Create full mixing coefficients array, setting unselected columns to 0
    mixing_coeffs = np.zeros(24)
    mixing_coeffs[selected_cols] = mixing_coeffs_selected

    os.makedirs(output_folder, exist_ok=True)

    heatmaps = []
    for i, bitplanes in enumerate(bitplanes_all_frames):
        # Weighted sum of the 24 bitplanes
        heatmap_float = np.zeros((H, W), dtype=np.float32)
        for j in range(24):
            heatmap_float += mixing_coeffs[j] * bitplanes[:, :, j].astype(np.float32)

        heatmap_norm = cv2.normalize(heatmap_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)

        heatmap_blurred = cv2.GaussianBlur(heatmap_uint8, (5, 5), 0)

        heatmap_colored = cv2.applyColorMap(heatmap_blurred, cv2.COLORMAP_JET)
        heatmaps.append(heatmap_colored)
    
    for i in range(len(heatmaps)):
        clip_num = (i // 4) + 1
        heatmap_num = (i % 4) + 1
        out_path = os.path.join(output_folder, f"clip_{clip_num}_heatmap_{heatmap_num}.png")
        cv2.imwrite(out_path, heatmaps[i])
        print(f"Saved: {out_path}")