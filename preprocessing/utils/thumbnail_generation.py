import os
import sys
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import VideoConfig

FRAME_SIZE = VideoConfig.FRAME_SIZE  
NUM_FRAMES_PER_CLIP = VideoConfig.NUM_FRAMES_PER_CLIP  # T
THUMBNAIL_SIZE = VideoConfig.THUMBNAIL_SIZE
GRID_ROWS = VideoConfig.GRID_ROWS
GRID_COLS = VideoConfig.GRID_COLS

def calculate_mask_bounds(H, W, mask_size, sub_h, sub_w, row, col):
    """Calculates valid mask bounds within sub-images, avoiding seams."""
    h_start = row * sub_h
    h_end = (row + 1) * sub_h
    w_start = col * sub_w
    w_end = (col + 1) * sub_w

    h = np.random.randint(h_start + mask_size // 2, h_end - mask_size // 2)
    w = np.random.randint(w_start + mask_size // 2, w_end - mask_size // 2)

    h1 = np.clip(h - mask_size // 2, h_start, h_end)
    h2 = np.clip(h + mask_size // 2, h_start, h_end)
    w1 = np.clip(w - mask_size // 2, w_start, w_end)
    w2 = np.clip(w + mask_size // 2, w_start, w_end)

    return h1, h2, w1, w2

def apply_mask(frames, mask_size, sub_h, sub_w):
    """Applies a consistent random mask to all frames within a clip."""
    H, W = frames.shape[1:3]  
    mask = np.ones((H, W), dtype=np.float32)

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            h1, h2, w1, w2 = calculate_mask_bounds(H, W, mask_size, sub_h, sub_w, row, col)
            mask[h1:h2, w1:w2] = 0  # Apply mask in the sub-image

    mask = tf.convert_to_tensor(mask, dtype=tf.float32) 
    mask = tf.expand_dims(mask, axis=-1)  # Add channel dimension
    mask = tf.expand_dims(mask, axis=0)  # Add batch dimension

    return frames * mask  

def generate_thumbnail(frames):
    """ Rearrange frames into grid layout, where each frame is resized to sub-image size (C × H/√t x W/√t)"""

    # Ensure that the total number of frames matches the expected grid size
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