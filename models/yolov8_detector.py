from ultralytics import YOLO
from pathlib import Path
import re
import cv2
import os
import tempfile

class YOLOv8Detector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Initialize YOLOv8 detector with specified model.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
        """
        self.model = YOLO(model_path)
    
    def detect_objects(self, video_path):
        """
        Detect objects in a video using YOLOv8.
        
        Args:
            video_path (str): Path to the input video
            
        Returns:
            list: List of dictionaries containing detection results:
                - class_name: Name of detected class
                - confidence: Detection confidence score
                - bbox: Tuple of (x, y, width, height)
                - frame_id: Frame number
        """
        # Create a temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Open the video file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            frame_count = 0
            detections = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame to temporary file
                frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Run inference on the frame
                results = self.model(frame_path)
                
                # Process results
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Get bounding box (convert from xyxy to xywh format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        width = x2 - x1
                        height = y2 - y1
                        
                        detections.append({
                            'class_name': class_name,
                            'confidence': confidence,
                            'bbox': (x1, y1, width, height),
                            'frame_id': frame_count
                        })
                
                frame_count += 1
            
            cap.release()
            return detections 