"""
Configuration module for the Flickd AI project.
Contains all configurable parameters and constants.
"""

from pathlib import Path

# Paths
DATA_DIR = Path("data")
TEMP_DIR = Path("temp")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

# YOLO Detection Config
YOLO_CONFIG = {
    "model_path": "yolov8n.pt",  # Using nano model for faster inference
    "conf_threshold": 0.3,  # Minimum confidence threshold
    "iou_threshold": 0.45,  # IoU threshold for NMS
    "frame_sample_rate": 5,  # Process every Nth frame
}

# CLIP + FAISS Matching Config
MATCHING_CONFIG = {
    "exact_threshold": 0.8,  # Cosine similarity threshold for exact matches
    "similar_threshold": 0.6,  # Threshold for similar matches
    "min_confidence": 0.5,  # Minimum confidence for valid matches
    "num_colors": 3,  # Number of dominant colors to extract
}

# Vibe Classification Config
VIBE_CONFIG = {
    "confidence_threshold": 0.4,  # Minimum confidence for vibe classification
    "top_k": 3,  # Number of top vibes to return
    "use_ensemble": True,  # Whether to use ensemble of models
    "temporal_window": 5,  # Number of frames to consider for temporal consistency
    "taxonomy": [
        "Coquette",
        "Clean Girl",
        "Cottagecore",
        "Streetcore",
        "Y2K",
        "Boho",
        "Party Glam",
        "Business Casual",
        "Athleisure",
        "Minimalist",
        "Vintage",
        "Evening",
        "Casual Chic",
        "Preppy",
        "Grunge"
    ]
}

# API Config
API_CONFIG = {
    "version": "v1",
    "rate_limit": 100,  # Requests per minute
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "allowed_extensions": [".mp4", ".avi", ".mov", ".mkv"],
}

# Fashion Items
FASHION_ITEMS = {
    'dress', 'handbag', 'tie', 'suitcase', 'umbrella', 'shoe', 'hat',
    'backpack', 'belt', 'suit', 'jacket', 'shirt', 'pants', 'skirt',
    'coat', 'sunglasses', 'scarf', 'boots', 'sneakers', 'jewelry'
}

# Create required directories
for directory in [DATA_DIR, TEMP_DIR, MODELS_DIR, OUTPUT_DIR]:
    directory.mkdir(exist_ok=True) 