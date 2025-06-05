import requests
import json
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_object_detection():
    """Test YOLOv8 object detection."""
    logger.info("Testing object detection...")
    try:
        from models.yolov8_detector import YOLOv8Detector
        detector = YOLOv8Detector()
        logger.info("✅ YOLOv8 detector initialized successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Object detection test failed: {str(e)}")
        return False

def test_product_matching():
    """Test CLIP + FAISS product matching."""
    logger.info("Testing product matching...")
    try:
        from models.clip_faiss_matcher import build_faiss_index_from_catalog
        faiss_index, product_ids = build_faiss_index_from_catalog('data/images.csv')
        logger.info("✅ FAISS index built successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Product matching test failed: {str(e)}")
        return False

def test_vibe_classification():
    """Test vibe classification."""
    logger.info("Testing vibe classification...")
    try:
        from vibe_classifier import VibeClassifier
        classifier = VibeClassifier()
        test_metadata = {
            "caption": "Summer outfit video",
            "description": "A casual summer look with floral dress",
            "tags": ["summer", "fashion", "outfit"]
        }
        vibes = classifier.classify_vibe(test_metadata)
        logger.info(f"✅ Vibe classification successful: {vibes}")
        return True
    except Exception as e:
        logger.error(f"❌ Vibe classification test failed: {str(e)}")
        return False

def test_api_endpoint():
    """Test API endpoint."""
    logger.info("Testing API endpoint...")
    try:
        # Test video path
        video_path = "data/videos/2025-05-22_08-25-12_UTC.mp4"
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Test video not found: {video_path}")

        # Prepare request
        files = {
            'video': ('video.mp4', open(video_path, 'rb')),
            'metadata': (None, json.dumps({
                'caption': 'Beyond The Curve',
                'description': 'A new standard, built from scratch',
                'tags': ['fashion', 'empowerment', 'style']
            }))
        }

        # Send request
        response = requests.post('http://localhost:8000/process-video', files=files)
        response.raise_for_status()
        
        # Check response format
        result = response.json()
        required_fields = ['video_id', 'vibes', 'products', 'processing_time']
        for field in required_fields:
            if field not in result:
                raise KeyError(f"Missing field in response: {field}")
        
        logger.info("✅ API endpoint test successful")
        return True
    except Exception as e:
        logger.error(f"❌ API endpoint test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    tests = [
        ("Object Detection", test_object_detection),
        ("Product Matching", test_product_matching),
        ("Vibe Classification", test_vibe_classification),
        ("API Endpoint", test_api_endpoint)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\nRunning {name} test...")
        success = test_func()
        results.append((name, success))
    
    # Print summary
    logger.info("\nTest Summary:")
    logger.info("=" * 50)
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{name}: {status}")
    logger.info("=" * 50)
    
    # Check if all tests passed
    all_passed = all(success for _, success in results)
    if all_passed:
        logger.info("\n🎉 All tests passed! The project meets all requirements.")
    else:
        logger.error("\n❌ Some tests failed. Please check the logs above for details.")

if __name__ == "__main__":
    main() 