import cv2

def get_image_channels(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return "Error: Could not load the image. Check the file path."
    
    shape = img.shape

    if len(shape) == 2:
        channels = 1 
    elif len(shape) == 3:
        channels = shape[2] 
    else:
        return "Error: Unexpected image format."
    
    return channels

image_path = r'C:\Users\Arfa Tech\Downloads\heatmap_01.jpg'  

result = get_image_channels(image_path)
if isinstance(result, int):
    print(f"The image has {result} channel(s).")
else:
    print(result)
