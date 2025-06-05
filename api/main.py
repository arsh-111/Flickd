import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import json

from models.yolov8_detector import YOLOv8Detector
from models.clip_faiss_matcher import build_faiss_index_from_catalog, get_clip_embedding, match_crop_to_catalog
from vibe_classifier import VibeClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flickd AI API",
    description="API for video processing with object detection, product matching, and vibe classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
try:
    detector = YOLOv8Detector()
    faiss_index, product_ids = build_faiss_index_from_catalog("data/images.csv")
    vibe_classifier = VibeClassifier()
    logger.info("All models initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

@app.get("/")
async def root():
    return {"message": "Welcome to Flickd AI API"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": {
            "object_detector": "initialized",
            "product_matcher": "initialized",
            "vibe_classifier": "initialized"
        }
    }

@app.post("/process-video")
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
            
            # Match products
            product_matches = []
            for det in detections:
                # For now, we'll just use the frame information
                # In a real implementation, you would crop the detected objects
                # and match them against the catalog
                product_matches.append({
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'frame_id': det['frame_id']
                })
            
            # Classify vibe (using dummy metadata for now)
            vibe = vibe_classifier.classify_vibe({"caption": video.filename})
            
            # Prepare response
            response = {
                "detections": detections,
                "product_matches": product_matches,
                "vibe": vibe
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