import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

mtcnn = MTCNN(keep_all=False, device=device)

def skin_segmentation(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    return skin_mask

def remove_non_skin_region(input_path, output_path):
    try:
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img)  
        
        boxes, probs = mtcnn.detect(img)
        
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]  
            x, y, w, h = box
            
            img_width, img_height = img.size
            left = max(0, int(x))
            top = max(0, int(y))
            right = min(img_width, int(x + w))
            bottom = min(img_height, int(y + h))
            
            face_region = img_np[top:bottom, left:right]
            skin_mask = skin_segmentation(face_region)
            skin_region = cv2.bitwise_and(face_region, face_region, mask=skin_mask)
            skin_region_pil = Image.fromarray(skin_region)
            skin_region_resized = skin_region_pil.resize((224, 224))

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            skin_region_resized.save(output_path)
            print(f"Saved skin region (224x224) to {output_path}")
            return True
        else:
            print(f"No face detected in {input_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False