# Flickd AI Hackathon

## Purpose

**Flickd** is an AI-powered video processing pipeline designed to analyze fashion-related videos. It automatically:
- Extracts frames from videos
- Detects fashion items using YOLOv8
- Crops detected objects
- Matches cropped objects to a product catalog using CLIP embeddings and FAISS similarity search
- Classifies the "vibe" or aesthetic of the video using a zero-shot classifier (e.g., Clean Girl, Y2K, Boho, etc.)
- Outputs structured results for further use

This project is ideal for fashion tech, e-commerce, and content analysis applications.

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/arsh-111/Flickd.git
cd Flickd
```

### 2. Set Up Python Environment

It's recommended to use a virtual environment:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

- Place your video files in `data/videos/` (sample videos are already provided).
- Ensure `data/images.csv` and `data/catalog.xlsx` are present (these are used for product matching).
- The pipeline will automatically create and use `data/frames/`, `data/crops/`, and other output folders.

### 5. Run the Main Pipeline

To process all videos and generate outputs:

```bash
python main.py
```

This will:
- Extract frames from each video
- Detect and crop fashion items
- Match items to the catalog
- Classify the vibe of each video
- Write results to output files

### 6. Run the API (Optional)

A FastAPI server is provided for programmatic access. You can start the server using:

```bash
python api/run_server.py
```

- Visit `http://127.0.0.1:8000/docs` for interactive API documentation.
- You can upload videos and get structured results via the `/process-video` endpoint.

---

## Output

- Cropped images: `data/crops/<video_id>/`
- Frame images: `data/frames/<video_id>/`
- Output results: (see output files or API response)

---

## How a Third Person Can Use This Project

1. **Clone the repository** as shown above.
2. **Install dependencies** in a virtual environment.
3. **Add your own videos** to `data/videos/`.
4. **Run the pipeline** with `python main.py` or use the API.
5. **Check the output** in the `data/` subfolders or via the API.

---

## Requirements

- Python 3.8+
- pip
- (Optional) CUDA-enabled GPU for faster inference

---

## Project Structure

- `main.py` — Main pipeline script
- `api/` — FastAPI server for programmatic access
- `models/` — Model wrappers and FAISS index
- `utils/` — Utility scripts (frame extraction, cropping, etc.)
- `data/` — Input videos, catalog, and outputs
- `requirements.txt` — Python dependencies

---

## Example

```bash
python main.py
```

or

```bash
uvicorn api.app:app --reload
```

---

## License

MIT
