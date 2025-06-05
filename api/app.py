from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import json
import os
from pathlib import Path
import shutil
import uuid
from datetime import datetime

# Import project components
from models.yolov8_detector import YOLOv8Detector
from models.clip_faiss_matcher import build_faiss_index_from_catalog, match_crop_to_catalog
from vibe_classifier import VibeClassifier
from utils.frame_extractor import extract_frames
from utils.object_cropper import crop_and_save

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flickd AI Video Processing API",
    description="API for processing fashion videos and detecting vibes and products",
    version="1.0.0"
)

# Initialize components
try:
    detector = YOLOv8Detector()
    vibe_classifier = VibeClassifier()
    faiss_index, product_ids = build_faiss_index_from_catalog('data/images.csv')
    logger.info("Successfully initialized all components")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# Define data models
class VideoMetadata(BaseModel):
    caption: Optional[str] = Field(None, description="Video caption")
    description: Optional[str] = Field(None, description="Detailed video description")
    tags: Optional[List[str]] = Field(None, description="List of video tags")

class ProductMatch(BaseModel):
    type: str = Field(..., description="Type of the product")
    color: str = Field(..., description="Color of the product")
    matched_product_id: str = Field(..., description="ID of the matched product")
    match_type: str = Field(..., description="Type of match (exact/similar/none)")
    confidence: float = Field(..., description="Confidence score of the match")

class VibeClassification(BaseModel):
    vibe: str = Field(..., description="Name of the vibe")
    confidence: float = Field(..., description="Confidence score of the classification")

class ProcessedVideoResponse(BaseModel):
    video_id: str = Field(..., description="Unique identifier for the video")
    vibes: List[VibeClassification] = Field(..., description="List of detected vibes")
    products: List[ProductMatch] = Field(..., description="List of matched products")
    processing_time: float = Field(..., description="Total processing time in seconds")

# Helper functions
def save_uploaded_video(file: UploadFile, video_id: str) -> Path:
    """Save uploaded video file to disk."""
    video_dir = Path("data/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    
    video_path = video_dir / f"{video_id}.mp4"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return video_path

def process_video_frames(video_path: Path, video_id: str) -> List[Dict]:
    """Process video frames and return product matches."""
    frames_dir = Path("data/frames") / video_id
    crops_dir = Path("data/crops") / video_id
    
    # Create directories
    frames_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract frames
    num_frames = extract_frames(
        video_path=str(video_path),
        out_dir=str(frames_dir),
        fps=1
    )
    
    matches = []
    for frame_path in frames_dir.glob("frame_*.jpg"):
        # Run detection
        detections = detector.detect_objects(str(frame_path))
        
        # Crop and save detected objects
        crop_paths = crop_and_save(
            image_path=str(frame_path),
            detections=detections,
            out_dir=str(crops_dir)
        )
        
        # Process each crop
        for crop_path in crop_paths:
            try:
                # Get crop embedding and match to catalog
                crop_embedding = get_clip_embedding(crop_path)
                matched_product_id, match_type, confidence = match_crop_to_catalog(
                    crop_embedding, faiss_index, product_ids
                )
                
                matches.append({
                    "type": "unknown",  # TODO: Add type detection
                    "color": "unknown",  # TODO: Add color detection
                    "matched_product_id": matched_product_id,
                    "match_type": match_type,
                    "confidence": confidence
                })
            except Exception as e:
                logger.error(f"Error processing crop {crop_path}: {str(e)}")
                continue
    
    return matches

# API endpoints
@app.post("/process-video", response_model=ProcessedVideoResponse)
async def process_video(
    video: UploadFile = File(...),
    metadata: Optional[VideoMetadata] = None
):
    """
    Process a video file and return detected vibes and matched products.
    
    Args:
        video: Uploaded video file
        metadata: Optional video metadata
        
    Returns:
        ProcessedVideoResponse: Processing results including vibes and products
    """
    start_time = datetime.now()
    video_id = str(uuid.uuid4())
    
    try:
        # Save uploaded video
        video_path = save_uploaded_video(video, video_id)
        logger.info(f"Saved video {video_id} to {video_path}")
        
        # Process video frames
        matches = process_video_frames(video_path, video_id)
        logger.info(f"Processed {len(matches)} product matches for video {video_id}")
        
        # Classify vibes
        if metadata:
            vibe_classifications = vibe_classifier.classify_vibe(metadata.dict())
        else:
            # Use empty metadata if none provided
            vibe_classifications = vibe_classifier.classify_vibe({})
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare response
        response = {
            "video_id": video_id,
            "vibes": vibe_classifications,
            "products": matches,
            "processing_time": processing_time
        }
        
        logger.info(f"Successfully processed video {video_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

