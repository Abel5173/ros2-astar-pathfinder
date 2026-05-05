"""
YOLO Detector Module

Provides a wrapper around Ultralytics YOLOv8 for object detection
with support for custom classes, confidence filtering, and performance optimization.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ultralytics import YOLO
import logging

# Suppress ultralytics logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)


class YOLODetector:
    """
    YOLOv8 Object Detector with performance optimizations.
    
    Handles model loading, inference, and result formatting for ROS 2 integration.
    """
    
    # COCO class names for YOLOv8
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(
        self,
        model_path: str = 'yolov8n.pt',
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        inference_size: int = 640,
        target_classes: Optional[List[str]] = None,
        device: str = 'auto'
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights or model name (e.g., 'yolov8n.pt')
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression IoU threshold
            inference_size: Input size for inference (default 640)
            target_classes: List of class names to detect (None = all classes)
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.inference_size = inference_size
        
        # Load model
        self.model = YOLO(model_path)
        
        # Determine device
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Build class filter
        self.target_class_ids = None
        if target_classes:
            self.target_class_ids = [
                self.COCO_CLASSES.index(cls) for cls in target_classes 
                if cls in self.COCO_CLASSES
            ]
        
        self.device = device
        
        # Warm up model
        self._warmup()
    
    def _warmup(self) -> None:
        """Run a dummy inference to warm up the model."""
        dummy_input = np.zeros((self.inference_size, self.inference_size, 3), dtype=np.uint8)
        self.model.predict(dummy_input, verbose=False)
    
    def detect(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Run object detection on an image.
        
        Args:
            image: OpenCV BGR image (H x W x 3)
            
        Returns:
            Tuple of (detections list, annotated image)
            
            Each detection dict contains:
                - bbox: [x1, y1, x2, y2] in pixel coordinates
                - class_id: int
                - class_name: str
                - confidence: float
                - center: (x, y) tuple
        """
        # Store original dimensions for scaling
        original_height, original_width = image.shape[:2]
        
        # Run inference
        results = self.model.predict(
            image,
            imgsz=self.inference_size,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            verbose=False,
            device=self.device,
            classes=self.target_class_ids
        )
        
        # Parse results
        detections = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f'class_{class_id}'
                
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'center': (float((x1 + x2) / 2), float((y1 + y2) / 2))
                }
                detections.append(detection)
        
        # Create annotated image
        annotated_image = self._annotate_image(image.copy(), detections)
        
        return detections, annotated_image
    
    def _annotate_image(
        self, 
        image: np.ndarray, 
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: OpenCV image
            detections: List of detection dictionaries
            
        Returns:
            Annotated image
        """
        # Generate colors for classes
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(self.COCO_CLASSES), 3), dtype=np.uint8)
        
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get color for this class
            color = tuple(int(c) for c in colors[class_id % len(colors)])
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f'{class_name}: {confidence:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y1, label_size[1] + 10)
            
            cv2.rectangle(
                image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        # Add detection count
        info_text = f'Detections: {len(detections)}'
        cv2.putText(
            image,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        return image
    
    def get_class_names(self) -> List[str]:
        """Return list of available class names."""
        return self.COCO_CLASSES.copy()
