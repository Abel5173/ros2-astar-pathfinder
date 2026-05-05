"""
Image Utilities Module

Helper functions for image processing, conversion, and visualization
for ROS 2 perception pipeline.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Header


def resize_image(
    image: np.ndarray, 
    target_size: Tuple[int, int],
    keep_aspect: bool = True
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to target size with optional aspect ratio preservation.
    
    Args:
        image: Input OpenCV image
        target_size: (width, height) target dimensions
        keep_aspect: If True, maintain aspect ratio with letterboxing
        
    Returns:
        Tuple of (resized image, scale factor, padding offset)
    """
    if not keep_aspect:
        resized = cv2.resize(image, target_size)
        return resized, 1.0, (0, 0)
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create letterboxed image
    padded = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # Calculate padding
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Place resized image in center
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return padded, scale, (pad_x, pad_y)


def restore_coordinates(
    bbox: Tuple[float, float, float, float],
    scale: float,
    padding: Tuple[int, int],
    original_size: Tuple[int, int]
) -> Tuple[float, float, float, float]:
    """
    Convert coordinates from resized image back to original image.
    
    Args:
        bbox: (x1, y1, x2, y2) in resized image coordinates
        scale: Scale factor used during resize
        padding: (pad_x, pad_y) padding applied
        original_size: (width, height) of original image
        
    Returns:
        bbox in original image coordinates
    """
    pad_x, pad_y = padding
    orig_w, orig_h = original_size
    
    # Remove padding and scale back
    x1 = (bbox[0] - pad_x) / scale
    y1 = (bbox[1] - pad_y) / scale
    x2 = (bbox[2] - pad_x) / scale
    y2 = (bbox[3] - pad_y) / scale
    
    # Clip to image bounds
    x1 = max(0, min(x1, orig_w))
    y1 = max(0, min(y1, orig_h))
    x2 = max(0, min(x2, orig_w))
    y2 = max(0, min(y2, orig_h))
    
    return (x1, y1, x2, y2)


def draw_detections(
    image: np.ndarray,
    detections: list,
    color_map: Optional[Dict[int, Tuple[int, int, int]]] = None
) -> np.ndarray:
    """
    Draw detection bounding boxes on image.
    
    Args:
        image: OpenCV image to draw on
        detections: List of detection dictionaries
        color_map: Optional dict mapping class_id to BGR color tuple
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    # Generate default colors if not provided
    if color_map is None:
        np.random.seed(42)
        num_classes = 80  # COCO classes
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
        color_map = {i: tuple(map(int, colors[i])) for i in range(num_classes)}
    
    for det in detections:
        bbox = det['bbox']
        class_id = det.get('class_id', 0)
        class_name = det.get('class_name', f'class_{class_id}')
        confidence = det.get('confidence', 0.0)
        
        color = color_map.get(class_id, (0, 255, 0))
        
        # Draw rectangle
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f'{class_name}: {confidence:.2f}'
        label_size, baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Label background
        label_y = max(y1, label_size[1] + 10)
        cv2.rectangle(
            annotated,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0], label_y),
            color,
            -1
        )
        
        # Label text
        cv2.putText(
            annotated,
            label,
            (x1, label_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return annotated


def create_detection_msg(
    detection: Dict[str, Any],
    header: Header
) -> Detection2D:
    """
    Convert detection dictionary to ROS 2 Detection2D message.
    
    Args:
        detection: Detection dictionary with bbox, class_id, confidence
        header: ROS message header
        
    Returns:
        Detection2D message
    """
    det_msg = Detection2D()
    det_msg.header = header
    
    # Extract bbox
    bbox = detection['bbox']
    x1, y1, x2, y2 = bbox
    
    # Set bounding box center and size
    det_msg.bbox.center.position.x = (x1 + x2) / 2.0
    det_msg.bbox.center.position.y = (y1 + y2) / 2.0
    det_msg.bbox.size_x = x2 - x1
    det_msg.bbox.size_y = y2 - y1
    
    # Add hypothesis with class and confidence
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = str(detection.get('class_id', 0))
    hypothesis.hypothesis.score = float(detection.get('confidence', 0.0))
    
    # Add class name as additional hypothesis
    class_name = detection.get('class_name', '')
    if class_name:
        name_hypothesis = ObjectHypothesisWithPose()
        name_hypothesis.hypothesis.class_id = f'name:{class_name}'
        name_hypothesis.hypothesis.score = 1.0
        det_msg.results.append(name_hypothesis)
    
    det_msg.results.append(hypothesis)
    
    return det_msg


def apply_nms(
    detections: list,
    iou_threshold: float = 0.5
) -> list:
    """
    Apply Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered list of detections
    """
    if not detections:
        return []
    
    # Sort by confidence
    sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while sorted_dets:
        current = sorted_dets.pop(0)
        keep.append(current)
        
        # Remove detections that overlap too much with current
        filtered = []
        for det in sorted_dets:
            iou = calculate_iou(current['bbox'], det['bbox'])
            if iou < iou_threshold:
                filtered.append(det)
        
        sorted_dets = filtered
    
    return keep


def calculate_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height
    
    # Calculate union
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area
