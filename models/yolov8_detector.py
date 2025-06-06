from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import tempfile
import numpy as np
from typing import List, Dict, Tuple
import logging
from config import YOLO_CONFIG, FASHION_ITEMS

logger = logging.getLogger(__name__)

class YOLOv8Detector:
    def __init__(self, model_path: str = YOLO_CONFIG["model_path"]):
        """
        Initialize YOLOv8 detector with specified model.
        
        Args:
            model_path (str): Path to YOLOv8 model weights
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = YOLO_CONFIG["conf_threshold"]
            self.iou_threshold = YOLO_CONFIG["iou_threshold"]
            self.frame_sample_rate = YOLO_CONFIG["frame_sample_rate"]
            logger.info(f"Initialized YOLOv8 detector with model: {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize YOLOv8 detector: {str(e)}")
            raise
    
    def apply_nms(self, boxes: List[Tuple], scores: List[float]) -> List[int]:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            boxes: List of (x1, y1, x2, y2) boxes
            scores: List of confidence scores
            
        Returns:
            List of indices to keep
        """
        x1 = np.array([box[0] for box in boxes])
        y1 = np.array([box[1] for box in boxes])
        x2 = np.array([box[2] for box in boxes])
        y2 = np.array([box[3] for box in boxes])
        
        areas = (x2 - x1) * (y2 - y1)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            
            if indices.size == 1:
                break
                
            # Compute IoU
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / (areas[indices[1:]] + areas[i] - w * h)
            
            inds = np.where(overlap <= self.iou_threshold)[0]
            indices = indices[inds + 1]
            
        return keep
    
    def detect_objects(self, video_path: str) -> List[Dict]:
        """
        Detect objects in a video using YOLOv8.
        
        Args:
            video_path (str): Path to the input video
            
        Returns:
            list: List of dictionaries containing detection results:
                - class_name: Name of detected class
                - confidence: Detection confidence score
                - bbox: Tuple of (x1, y1, x2, y2)
                - frame_id: Frame number
        """
        try:
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
                    
                    # Sample frames
                    if frame_count % self.frame_sample_rate != 0:
                        frame_count += 1
                        continue
                    
                    # Save frame to temporary file
                    frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Run inference on the frame
                    results = self.model(frame_path, conf=self.conf_threshold)
                    
                    # Process results
                    for result in results:
                        boxes = result.boxes
                        
                        # Prepare boxes and scores for NMS
                        box_coords = []
                        scores = []
                        classes = []
                        
                        for box in boxes:
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]
                            
                            # Skip if not a fashion item
                            if class_name.lower() not in FASHION_ITEMS:
                                continue
                                
                            confidence = float(box.conf[0])
                            if confidence < self.conf_threshold:
                                continue
                                
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            box_coords.append((x1, y1, x2, y2))
                            scores.append(confidence)
                            classes.append(class_name)
                        
                        # Apply NMS if we have detections
                        if box_coords:
                            keep_indices = self.apply_nms(box_coords, scores)
                            
                            # Add kept detections
                            for idx in keep_indices:
                                x1, y1, x2, y2 = box_coords[idx]
                                detections.append({
                                    'class_name': classes[idx],
                                    'confidence': scores[idx],
                                    'bbox': (x1, y1, x2 - x1, y2 - y1),  # Convert to (x, y, w, h)
                                    'frame_id': frame_count
                                })
                    
                    frame_count += 1
                    
                    # Log progress
                    if frame_count % 50 == 0:
                        logger.info(f"Processed {frame_count} frames...")
                
                cap.release()
                logger.info(f"Detection complete. Found {len(detections)} objects in {frame_count} frames")
                return detections
                
        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            raise 