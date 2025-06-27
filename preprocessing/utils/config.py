import numpy as np

class VideoConfig:
    MASK_SIZE = 50 
    FRAME_SIZE = 224 
    NUM_FRAMES_PER_CLIP = 4 
    NUM_CLIPS = 8 
    THUMBNAIL_SIZE = (224, 224)  
    GRID_ROWS = int(np.ceil(np.sqrt(NUM_FRAMES_PER_CLIP)))
    GRID_COLS = int(np.ceil(NUM_FRAMES_PER_CLIP / GRID_ROWS))

    FRAME_RATE = 30  
    LOWCUT = 0.7
    HIGHCUT = 2.0