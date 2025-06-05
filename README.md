# Flickd AI Video Processing System

An AI-powered system for analyzing fashion videos, detecting products, and classifying aesthetic vibes. The system uses YOLOv8 for object detection, CLIP + FAISS for product matching, and transformer-based models for vibe classification.

## Features

- **Object Detection**: Detects fashion items in video frames using YOLOv8
- **Product Matching**: Matches detected items to a product catalog using CLIP embeddings and FAISS
- **Vibe Classification**: Classifies video aesthetic using transformer-based models
- **REST API**: FastAPI-based API for video processing and analysis
- **Comprehensive Logging**: Detailed logging of all operations and errors

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/arsh-111/Flickd.git
cd flickd-ai
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required models:
```bash
# YOLOv8 model will be downloaded automatically on first run
# CLIP model will be downloaded automatically on first run
```

### Directory Structure

```
flickd-ai/
├── api/
│   ├── app.py           # FastAPI application
│   └── run_server.py    # Server runner script
├── data/
│   ├── videos/          # Uploaded videos
│   ├── frames/          # Extracted video frames
│   ├── crops/           # Cropped fashion items
│   └── metadata/        # Video metadata files
├── models/
│   ├── yolov8_detector.py
│   └── clip_faiss_matcher.py
├── utils/
│   ├── frame_extractor.py
│   └── object_cropper.py
├── vibe_classifier.py
├── main.py
└── requirements.txt
```

## Usage

### Running the API Server

1. Start the FastAPI server:
```bash
python api/run_server.py
```

The server will start at `http://localhost:8000`. You can access the API documentation at `http://localhost:8000/docs`.

### Processing a Video

#### Using Python

```python
import requests
import json

# Prepare the request
files = {
    'video': ('video.mp4', open('video.mp4', 'rb')),
    'metadata': (None, json.dumps({
        'caption': 'Summer outfit video',
        'description': 'A casual summer look with floral dress and straw hat',
        'tags': ['summer', 'fashion', 'outfit', 'floral', 'dress']
    }))
}

# Send the request
response = requests.post('http://localhost:8000/process-video', files=files)

# Process the response
result = response.json()
print(f"Video ID: {result['video_id']}")
print(f"Detected Vibes: {result['vibes']}")
print(f"Matched Products: {result['products']}")
print(f"Processing Time: {result['processing_time']} seconds")
```

#### Using cURL

```bash
curl -X POST "http://localhost:8000/process-video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@video.mp4" \
  -F 'metadata={"caption":"Summer outfit","description":"A casual summer look","tags":["summer","fashion"]}'
```

## API Documentation

### Endpoints

#### POST /process-video

Process a video file and return detected vibes and matched products.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameters:
  - `video`: Video file (required)
  - `metadata`: JSON string containing video metadata (optional)
    ```json
    {
      "caption": "string",
      "description": "string",
      "tags": ["string"]
    }
    ```

**Response:**
```json
{
  "video_id": "string",
  "vibes": [
    {
      "vibe": "string",
      "confidence": 0.0
    }
  ],
  "products": [
    {
      "type": "string",
      "color": "string",
      "matched_product_id": "string",
      "match_type": "string",
      "confidence": 0.0
    }
  ],
  "processing_time": 0.0
}
```

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Demo

[Watch the system demo on Loom](https://www.loom.com/share/your-demo-link)

The demo covers:
- System setup and installation
- Processing a sample video
- Understanding the API responses
- Real-time product matching and vibe classification

## Supported Vibe Categories

The system can classify videos into the following aesthetic vibes:
- Coquette
- Clean Girl
- Cottagecore
- Streetcore
- Y2K
- Boho
- Party Glam

## Error Handling

The API includes comprehensive error handling:
- Invalid file formats
- Missing required parameters
- Processing errors
- Server errors

All errors are logged and returned with appropriate HTTP status codes and error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 for object detection
- OpenAI CLIP for image embeddings
- FAISS for efficient similarity search
- FastAPI for the web framework
