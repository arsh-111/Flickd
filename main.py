import os
from pathlib import Path
from utils.frame_extractor import extract_frames
from models.yolov8_detector import YOLOv8Detector
from utils.object_cropper import crop_and_save
from models.clip_faiss_matcher import get_clip_embedding, build_faiss_index_from_catalog, match_crop_to_catalog
from pipeline.output_writer import write_output
from vibe_classifier import VibeClassifier
import json

def get_video_metadata(video_id: str) -> dict:
    """
    Get metadata for a video from its metadata file.
    
    Args:
        video_id (str): ID of the video
        
    Returns:
        dict: Video metadata including caption, description, and tags
    """
    metadata_path = Path(f"data/metadata/{video_id}.json")
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata for {video_id}: {e}")
    return {}

def process_video(video_id: str, faiss_index, product_ids, vibe_classifier: VibeClassifier):
    """Process all crops for a video and return matches."""
    crop_dir = f"data/crops/{video_id}"
    matches = []
    
    if not os.path.exists(crop_dir):
        print(f"No crops found for video {video_id}")
        return matches
    
    for crop_file in os.listdir(crop_dir):
        crop_path = os.path.join(crop_dir, crop_file)
        
        try:
            # Get crop embedding
            crop_embedding = get_clip_embedding(crop_path)
            
            # Match to catalog
            matched_product_id, match_type, confidence = match_crop_to_catalog(
                crop_embedding, faiss_index, product_ids
            )
            
            # Add match to results
            matches.append({
                "type": "unknown",  # TODO: Add type detection
                "color": "unknown",  # TODO: Add color detection
                "matched_product_id": matched_product_id,
                "match_type": match_type,
                "confidence": confidence
            })
            
            print(f"Matched {crop_file} to {matched_product_id} ({match_type}, conf: {confidence:.2f})")
            
        except Exception as e:
            print(f"Error processing {crop_file}: {e}")
            continue
    
    return matches

def process_videos():
    """
    Process all MP4 videos in the data/videos directory and extract frames.
    """
    # Define paths
    videos_dir = Path("data/videos")
    frames_dir = Path("data/frames")
    crops_dir = Path("data/crops")
    catalog_xlsx = Path("data/catalog.xlsx")
    
    # Create directories if they don't exist
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing YOLOv8 detector...")
    # Initialize YOLOv8 detector
    detector = YOLOv8Detector()
    print("YOLOv8 detector initialized successfully")
    
    # Process each video file
    for video_path in videos_dir.glob("*.mp4"):
        print(f"\nProcessing video: {video_path.name}")
        # Get video ID from filename (without extension)
        video_id = video_path.stem
        
        # Create output directories for this video
        video_frames_dir = frames_dir / video_id
        video_crops_dir = crops_dir / video_id
        video_frames_dir.mkdir(exist_ok=True)
        video_crops_dir.mkdir(exist_ok=True)
        
        try:
            # Extract frames
            num_frames = extract_frames(
                video_path=str(video_path),
                out_dir=str(video_frames_dir),
                fps=1
            )
            print(f"Extracted {num_frames} frames for {video_id}")
            
            # Process each frame with YOLOv8
            frame_count = 0
            for frame_path in video_frames_dir.glob("frame_*.jpg"):
                frame_count += 1
                print(f"\nProcessing frame {frame_count}/{num_frames}: {frame_path.name}")
                
                # Run detection
                print("Running YOLO detection...")
                detections = detector.detect_objects(str(frame_path))
                print(f"Found {len(detections)} total detections")
                
                # Print detections for this frame
                print("Detections:")
                for det in detections:
                    print(f"  - {det['class_name']} (conf: {det['confidence']:.2f}) at {det['bbox']}")
                
                # Crop and save detected objects
                print("Cropping and saving objects...")
                crop_paths = crop_and_save(
                    image_path=str(frame_path),
                    detections=detections,
                    out_dir=str(video_crops_dir)
                )
                
                # Print saved crops
                if crop_paths:
                    print(f"Saved {len(crop_paths)} crops:")
                    for crop_path in crop_paths:
                        print(f"  - {Path(crop_path).name}")
                else:
                    print("No fashion items detected in this frame")
            
        except Exception as e:
            print(f"Error processing {video_id}: {str(e)}")

def main():
    # Initialize vibe classifier
    print("Initializing vibe classifier...")
    vibe_classifier = VibeClassifier()
    print("Vibe classifier initialized successfully")
    
    # Build or load FAISS index
    faiss_index, product_ids = build_faiss_index_from_catalog('data/images.csv')
    
    # Process each video directory
    for video_id in os.listdir('data/crops'):
        print(f"\nProcessing video: {video_id}")
        
        # Get matches for this video
        matches = process_video(video_id, faiss_index, product_ids, vibe_classifier)
        
        if matches:
            # Get video metadata
            video_metadata = get_video_metadata(video_id)
            
            # Classify vibes
            vibe_classifications = vibe_classifier.classify_vibe(video_metadata)
            
            # Extract vibe names from classifications
            vibes = [classification["vibe"] for classification in vibe_classifications]
            
            # Write results
            write_output(video_id, matches, vibes)
        else:
            print(f"No matches found for video {video_id}")

if __name__ == "__main__":
    process_videos()
    main()
