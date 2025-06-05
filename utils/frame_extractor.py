import cv2
import os
from pathlib import Path

def extract_frames(video_path, out_dir, fps=1):
    """
    Extract frames from a video at specified FPS and save them as JPEGs.
    
    Args:
        video_path (str): Path to the input video file
        out_dir (str): Directory to save extracted frames
        fps (int): Number of frames to extract per second (default: 1)
    
    Returns:
        int: Number of frames extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval based on desired FPS
    frame_interval = max(1, int(video_fps / fps))
    
    # Extract frames
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame if it's at the desired interval
        if frame_idx % frame_interval == 0:
            output_path = os.path.join(out_dir, f"frame_{saved_count}.jpg")
            cv2.imwrite(output_path, frame)
            saved_count += 1
            
        frame_idx += 1
    
    # Release resources
    cap.release()
    
    return saved_count
