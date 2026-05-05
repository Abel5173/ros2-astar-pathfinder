"""
Perception Node - ROS 2 Object Detection using YOLOv8

This module implements a ROS 2 node that subscribes to camera images,
performs object detection using YOLOv8, and publishes detection results.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import time

from .utils.detector import YOLODetector
from .utils.image_utils import resize_image, draw_detections, create_detection_msg


class PerceptionNode(Node):
    """
    ROS 2 Perception Node for real-time object detection.
    
    Subscribes to camera topics, runs YOLOv8 inference, and publishes
    detection results with bounding boxes.
    """
    
    def __init__(self) -> None:
        """Initialize the perception node with parameters and subscriptions."""
        super().__init__('perception_node')
        
        # Declare parameters individually with explicit types to avoid initialization issues
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('detection_topic', '/camera/detections')
        self.declare_parameter('visualization_topic', '/camera/detections/visualization')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('inference_size', 640)
        self.declare_parameter('enable_depth', True)
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('use_compressed', False)
        
        # Declare target_classes with explicit type - THIS IS THE KEY FIX
        self.declare_parameter('target_classes', Parameter.Type.STRING_ARRAY)
        
        # Get parameter values
        self.input_topic = self.get_parameter('input_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.visualization_topic = self.get_parameter('visualization_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.inference_size = self.get_parameter('inference_size').value
        self.enable_depth = self.get_parameter('enable_depth').value
        self.publish_visualization = self.get_parameter('publish_visualization').value
        self.use_compressed = self.get_parameter('use_compressed').value
        
        # Handle target_classes with safety check
        try:
            target_param = self.get_parameter('target_classes')
            if target_param.type_ == Parameter.Type.NOT_SET:
                self.target_classes = []
                self.get_logger().debug('target_classes not set, using empty list (all classes)')
            else:
                val = target_param.value
                self.target_classes = val if val is not None else []
        except Exception as e:
            self.get_logger().warn(f'Parameter target_classes issue: {e}. Using empty list.')
            self.target_classes = []
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Initialize YOLO detector
        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        try:
            self.detector = YOLODetector(
                model_path=self.model_path,
                confidence_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
                inference_size=self.inference_size,
                target_classes=self.target_classes if self.target_classes else None
            )
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {str(e)}')
            raise
        
        # Setup QoS profile
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create subscriptions
        self._setup_subscriptions()
        
        # Create publishers
        self._setup_publishers()
        
        # Initialize storage for synchronized messages
        self.latest_color_image: Optional[np.ndarray] = None
        self.latest_depth_image: Optional[np.ndarray] = None
        self.camera_info: Optional[CameraInfo] = None
        self.image_header: Optional[Header] = None
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0.0
        
        self.get_logger().info('Perception node initialized successfully')
        self.get_logger().info(f'Input topic: {self.input_topic}')
        self.get_logger().info(f'Output topic: {self.detection_topic}')
        self.get_logger().info(f'Target classes: {self.target_classes if self.target_classes else "all"}')
    
    def _setup_subscriptions(self) -> None:
        """Setup ROS 2 subscribers for camera topics."""
        # Color image subscription
        if self.use_compressed:
            self.color_sub = self.create_subscription(
                CompressedImage,
                self.input_topic + '/compressed',
                self._compressed_image_callback,
                self.qos_profile
            )
        else:
            self.color_sub = self.create_subscription(
                Image,
                self.input_topic,
                self._image_callback,
                self.qos_profile
            )
        
        # Depth image subscription (optional)
        if self.enable_depth:
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_topic,
                self._depth_callback,
                self.qos_profile
            )
            self.get_logger().info(f'Depth subscription enabled: {self.depth_topic}')
        
        # Camera info subscription
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            self.qos_profile
        )
    
    def _setup_publishers(self) -> None:
        """Setup ROS 2 publishers for detection results."""
        # Detection results publisher
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.detection_topic,
            10
        )
        
        # Visualization publisher (optional)
        if self.publish_visualization:
            self.vis_pub = self.create_publisher(
                Image,
                self.visualization_topic,
                10
            )
    
    def _camera_info_callback(self, msg: CameraInfo) -> None:
        """Store camera calibration information."""
        self.camera_info = msg
    
    def _depth_callback(self, msg: Image) -> None:
        """Handle incoming depth images."""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().warn(f'Failed to convert depth image: {str(e)}')
    
    def _compressed_image_callback(self, msg: CompressedImage) -> None:
        """Handle incoming compressed color images."""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self._process_image(cv_image, msg.header)
        except Exception as e:
            self.get_logger().error(f'Failed to process compressed image: {str(e)}')
    
    def _image_callback(self, msg: Image) -> None:
        """Handle incoming color images from ROS Image message."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self._process_image(cv_image, msg.header)
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'Unexpected error in image callback: {str(e)}')
    
    def _process_image(self, cv_image: np.ndarray, header: Header) -> None:
        """
        Process image through YOLO detection pipeline.
        
        Args:
            cv_image: OpenCV BGR image
            header: ROS message header for timestamp/frame_id
        """
        self.image_header = header
        self.latest_color_image = cv_image
        
        start_time = time.time()
        
        try:
            # Run YOLO inference
            detections, annotated_image = self.detector.detect(cv_image)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.total_inference_time += inference_time
            self.frame_count += 1
            
            # Log performance periodically
            if self.frame_count % 30 == 0:
                avg_time = self.total_inference_time / self.frame_count
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(
                    f'Performance - Avg inference: {avg_time*1000:.1f}ms, '
                    f'FPS: {fps:.1f}, Detections: {len(detections)}'
                )
            
            # Publish detection results
            self._publish_detections(detections, header)
            
            # Publish visualization
            if self.publish_visualization:
                self._publish_visualization(annotated_image, header)
                
        except Exception as e:
            self.get_logger().error(f'Error during inference: {str(e)}')
    
    def _publish_detections(self, detections: List[Dict[str, Any]], header: Header) -> None:
        """
        Publish detection results as Detection2DArray message.
        
        Args:
            detections: List of detection dictionaries with bbox, class_id, confidence
            header: ROS message header
        """
        detection_array = Detection2DArray()
        detection_array.header = header
        
        for det in detections:
            detection_msg = create_detection_msg(det, header)
            
            # Add depth information if available
            if self.enable_depth and self.latest_depth_image is not None:
                depth_value = self._get_depth_at_detection(det)
                if depth_value is not None:
                    # Store depth as additional hypothesis score
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = 'depth_m'
                    hypothesis.hypothesis.score = float(depth_value)
                    detection_msg.results.append(hypothesis)
            
            detection_array.detections.append(detection_msg)
        
        self.detection_pub.publish(detection_array)
    
    def _get_depth_at_detection(self, detection: Dict[str, Any]) -> Optional[float]:
        """
        Extract depth value at the center of a detection.
        
        Args:
            detection: Detection dictionary with bounding box
            
        Returns:
            Depth value in meters or None if unavailable
        """
        if self.latest_depth_image is None:
            return None
        
        try:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Ensure coordinates are within image bounds
            h, w = self.latest_depth_image.shape[:2]
            center_x = max(0, min(center_x, w - 1))
            center_y = max(0, min(center_y, h - 1))
            
            depth_value = self.latest_depth_image[center_y, center_x]
            
            # Handle different depth encodings
            if np.isnan(depth_value) or depth_value <= 0:
                return None
                
            # Convert to meters if needed (assuming millimeters)
            if depth_value > 1000:
                depth_value = depth_value / 1000.0
                
            return float(depth_value)
            
        except Exception as e:
            self.get_logger().warn(f'Failed to get depth: {str(e)}')
            return None
    
    def _publish_visualization(self, annotated_image: np.ndarray, header: Header) -> None:
        """
        Publish annotated image with bounding boxes.
        
        Args:
            annotated_image: OpenCV image with drawn detections
            header: ROS message header
        """
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
            vis_msg.header = header
            self.vis_pub.publish(vis_msg)
        except CvBridgeError as e:
            self.get_logger().warn(f'Failed to publish visualization: {str(e)}')


def main(args=None):
    """Main entry point for the perception node."""
    rclpy.init(args=args)
    
    try:
        node = PerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('Perception node stopped by user')
    except Exception as e:
        print(f'Error running perception node: {str(e)}')
    finally:
        # Cleanup
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()