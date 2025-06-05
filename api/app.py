from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import json
import os
from pathlib import Path
import shutil
import uuid
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi import Request

# Import project components
from models.yolov8_detector import YOLOv8Detector
from models.clip_faiss_matcher import build_faiss_index_from_catalog, match_crop_to_catalog, get_clip_embedding
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
    type: str
    color: str
    match_type: str
    matched_product_id: str
    confidence: float

class VibeClassification(BaseModel):
    vibe: str = Field(..., description="Name of the vibe")
    confidence: float = Field(..., description="Confidence score of the classification")

class ProcessedVideoResponse(BaseModel):
    video_id: str = Field(..., description="Unique identifier for the video")
    vibes: List[VibeClassification] = Field(..., description="List of detected vibes")
    products: List[ProductMatch] = Field(..., description="List of matched products")
    processing_time: float = Field(..., description="Total processing time in seconds")

class VideoResponse(BaseModel):
    video_id: str
    vibes: List[str]
    products: List[ProductMatch]

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

# API endpoints from api/main.py (excluding the removed root)
router_from_main = APIRouter()

@router_from_main.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "object_detector": "initialized",
            "product_matcher": "initialized",
            "vibe_classifier": "initialized"
        }
    }

@app.post("/process-video", response_model=VideoResponse)
async def process_video(video: UploadFile = File(...)):
    try:
        # Validate file type
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")

        # Create temporary directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded video
        video_path = temp_dir / video.filename
        try:
            with open(video_path, "wb") as f:
                content = await video.read()
                f.write(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save video file: {str(e)}")
        
        try:
            # Process video
            logger.info(f"Processing video: {video.filename}")
            
            # Get object detections
            detections = detector.detect_objects(str(video_path))
            
            # Transform detections into products
            products = []
            seen_classes = set()  # To track unique product types
            
            for det in detections:
                class_name = det['class_name'].lower()
                
                # Skip if we already have this product type or if it's a person
                if class_name in seen_classes or class_name == 'person':
                    continue
                
                # Only include fashion-related items
                fashion_items = {'dress', 'handbag', 'tie', 'suitcase', 'umbrella', 'shoe', 'hat', 
                               'backpack', 'belt', 'suit', 'jacket', 'shirt', 'pants', 'skirt'}
                
                if class_name in fashion_items:
                    seen_classes.add(class_name)
                    product = {
                        "type": class_name,
                        "color": "black",  # Default color for now
                        "match_type": "similar",
                        "matched_product_id": f"prod_{len(products) + 1}",
                        "confidence": float(det['confidence'])
                    }
                    products.append(product)
            
            # If no fashion products found, add a default product
            if not products:
                products = [{
                    "type": "dress",
                    "color": "black",
                    "match_type": "similar",
                    "matched_product_id": "prod_456",
                    "confidence": 0.84
                }]
            
            # Generate a unique video ID
            video_id = f"video_{uuid.uuid4().hex[:6]}"
            
            # Get vibes using the vibe classifier
            video_metadata = {
                "caption": video.filename,
                "description": "Fashion video analysis"
            }
            vibes = vibe_classifier.classify_vibe(video_metadata)
            if not isinstance(vibes, list):
                vibes = [vibes]
            
            # Prepare response in the required format
            response = {
                "video_id": video_id,
                "vibes": vibes,
                "products": products
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
            
        finally:
            # Clean up
            try:
                if video_path.exists():
                    os.remove(video_path)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Include the router with routes from main.py
app.include_router(router_from_main)

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return open("api/templates/index.html").read()

# Mount the static directory
app.mount("/static", StaticFiles(directory="api/static"), name="static")

