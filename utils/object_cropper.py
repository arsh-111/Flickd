import cv2
import os
from pathlib import Path

def crop_and_save(image_path, detections, out_dir, allowed_classes=["dress", "top", "jacket", "earrings", "bag", "shoes", "person", "tie", "handbag", "backpack", "umbrella", "hat", "helmet", "glasses", "sunglasses", "watch", "necklace", "bracelet", "ring"]):
    """
    Crop detected objects from an image and save them as separate files.
    
    Args:
        image_path (str): Path to the input image
        detections (list): List of detection dictionaries
        out_dir (str): Directory to save cropped images
        allowed_classes (list, optional): List of class names to keep
        
    Returns:
        list: Paths to saved crop files
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    saved_paths = []
    
    # Process each detection
    for idx, det in enumerate(detections):
        # Skip if class not in allowed_classes
        if det['class_name'] not in allowed_classes:
            continue
            
        # Get bounding box
        x, y, w, h = map(int, det['bbox'])
        
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        # Crop image
        crop = image[y:y+h, x:x+w]
        
        # Generate output filename
        frame_id = det['frame_id']
        class_name = det['class_name']
        output_path = os.path.join(out_dir, f"crop_{frame_id}_{idx}_{class_name}.jpg")
        
        # Save crop
        cv2.imwrite(output_path, crop)
        saved_paths.append(output_path)
    
    return saved_paths
