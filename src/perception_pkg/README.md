# ROS 2 Perception Pipeline with YOLOv8

A production-quality real-time object detection system using ROS 2 Humble, YOLOv8, and OpenCV. This package subscribes to camera topics from Gazebo simulations, processes images using YOLOv8, and publishes detection results.

## Features

- **Real-time Object Detection**: YOLOv8 integration with GPU/CPU support
- **ROS 2 Integration**: Full compatibility with ROS 2 Humble
- **Depth Estimation**: Optional depth integration for 3D positioning
- **Visualization**: Annotated images with bounding boxes and labels
- **Performance Optimized**: Resize preprocessing, efficient inference
- **Modular Architecture**: Clean separation of concerns
- **Configurable**: YAML-based parameter configuration
- **Multi-class Support**: All 80 COCO classes, with optional filtering

## Project Structure

```
perception_pkg/
├── perception_pkg/              # Python package
│   ├── __init__.py
│   ├── perception_node.py       # Main ROS 2 node
│   └── utils/
│       ├── __init__.py
│       ├── detector.py          # YOLOv8 wrapper
│       └── image_utils.py       # Image processing utilities
├── launch/
│   └── perception.launch.py     # Launch file
├── config/
│   ├── params.yaml              # Parameter configuration
│   └── rviz_config.rviz         # RViz configuration
├── resource/
│   └── perception_pkg           # Package marker
├── package.xml                  # ROS 2 package manifest
├── setup.py                     # Python package setup
├── setup.cfg                    # Setup configuration
└── README.md                    # This file
```

## Prerequisites

- ROS 2 Humble Hawksbill
- Python 3.8+
- OpenCV 4.7+
- CUDA (optional, for GPU acceleration)
- Gazebo (for simulation)

## Step-by-Step Installation

### Step 1: Create Workspace

```bash
# Create a new ROS 2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# Copy or clone the perception_pkg into src/
# (Assuming you have the package files)
```

### Step 2: Install Python Dependencies

```bash
# Install ultralytics (YOLOv8)
pip3 install ultralytics

# Install additional dependencies
pip3 install opencv-python numpy
```

### Step 3: Install ROS 2 Dependencies

```bash
cd ~/ros2_ws

# Install ROS 2 package dependencies
rosdep install --from-paths src --ignore-src -r -y

# If rosdep is not initialized:
# sudo rosdep init
# rosdep update
```

### Step 4: Build the Workspace

```bash
cd ~/ros2_ws

# Build the package
colcon build --packages-select perception_pkg --symlink-install

# For full build:
# colcon build --symlink-install
```

**Note**: The `--symlink-install` flag allows you to edit Python files without rebuilding.

### Step 5: Source the Environment

```bash
# Source the workspace (run this in every new terminal)
source ~/ros2_ws/install/setup.bash

# Or add to ~/.bashrc for automatic sourcing:
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Running the Perception Pipeline

### Method 1: Run Node Directly

```bash
# Source the workspace first
source ~/ros2_ws/install/setup.bash

# Run the perception node with default parameters
ros2 run perception_pkg perception_node

# Run with custom parameters
ros2 run perception_pkg perception_node \
  --ros-args \
  -p input_topic:=/camera/image_raw \
  -p model_path:=yolov8n.pt \
  -p confidence_threshold:=0.6
```

### Method 2: Launch File (Recommended)

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# Launch with default parameters
ros2 launch perception_pkg perception.launch.py

# Launch with custom parameters file
ros2 launch perception_pkg perception.launch.py \
  params_file:=/path/to/custom_params.yaml
```

## Verification Commands

### Check Available Topics

```bash
# List all active topics
ros2 topic list

# Expected output:
# /camera/camera_info
# /camera/detections
# /camera/detections/visualization
# /camera/image_raw
# /parameter_events
# /rosout
```

### Monitor Detection Output

```bash
# Echo detection messages (text output)
ros2 topic echo /camera/detections

# Monitor detection topic at 1Hz
ros2 topic hz /camera/detections

# Show bandwidth usage
ros2 topic bw /camera/detections
```

### Check Node Status

```bash
# List all nodes
ros2 node list

# Expected output:
# /perception_node

# Show node info
ros2 node info /perception_node
```

### Verify Image Topics

```bash
# Show image topic info
ros2 topic info /camera/image_raw
ros2 topic info /camera/detections/visualization

# Echo compressed image (if using compressed transport)
ros2 topic echo /camera/image_raw/compressed
```

## RViz Integration

### Step 1: Start RViz

```bash
# Source workspace
source ~/ros2_ws/install/setup.bash

# Launch RViz with default configuration
rviz2

# Or launch with provided configuration
rviz2 -d ~/ros2_ws/src/perception_pkg/config/rviz_config.rviz
```

### Step 2: Configure Displays

#### View Raw Camera Feed:
1. Click **Add** button (bottom left)
2. Select **Image** from the list
3. In the Display panel, set **Image Topic** to `/camera/image_raw`
4. Set **Transport Hint** to `raw`

#### View Detection Output:
1. Click **Add** button
2. Select **Image**
3. Set **Image Topic** to `/camera/detections/visualization`
4. Position it below or beside the raw camera feed

#### Save Configuration:
1. Go to **File > Save Config As**
2. Save to `~/ros2_ws/src/perception_pkg/config/rviz_config.rviz`

### RViz Keyboard Shortcuts

- **F**: Focus on selected display
- **G**: Move camera to look at point
- **R**: Reset view
- **Ctrl+S**: Save configuration

## Configuration

### Parameter File (params.yaml)

```yaml
# Input/Output Topics
input_topic: "/camera/image_raw"
depth_topic: "/camera/depth/image_raw"
camera_info_topic: "/camera/camera_info"
detection_topic: "/camera/detections"
visualization_topic: "/camera/detections/visualization"

# Model Configuration
model_path: "yolov8n.pt"  # n=Nano, s=Small, m=Medium, l=Large, x=Extra Large
confidence_threshold: 0.5
nms_threshold: 0.45
inference_size: 640

# Feature Flags
enable_depth: true
publish_visualization: true
use_compressed: false

# Target Classes (filter detections)
target_classes: []  # Empty = all classes
# Example: ["person", "car", "dog"]
```

### Model Options

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| yolov8n.pt | Fastest | Lower | Real-time, edge devices |
| yolov8s.pt | Fast | Medium | Balance speed/accuracy |
| yolov8m.pt | Medium | Good | General purpose |
| yolov8l.pt | Slower | Better | High accuracy needs |
| yolov8x.pt | Slowest | Best | Maximum accuracy |

## Gazebo Simulation Setup

### Launch Gazebo with Camera

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Launch Gazebo with a camera-equipped robot
ros2 launch gazebo_ros gazebo.launch.py

# Or use a specific robot with camera
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Verify Camera Topics

```bash
# Check if camera topics are publishing
ros2 topic list | grep camera
ros2 topic hz /camera/image_raw
```

## Troubleshooting

### Common Issues

#### Issue: `ModuleNotFoundError: No module named 'ultralytics'`
**Solution:**
```bash
pip3 install ultralytics
```

#### Issue: `cv_bridge` not found
**Solution:**
```bash
sudo apt install ros-humble-cv-bridge
```

#### Issue: `vision_msgs` not found
**Solution:**
```bash
sudo apt install ros-humble-vision-msgs
```

#### Issue: Slow inference speed
**Solutions:**
1. Use smaller model: `yolov8n.pt`
2. Reduce `inference_size` to 320 or 416
3. Enable GPU: Check CUDA installation
4. Reduce `confidence_threshold` to filter earlier

#### Issue: No detections shown
**Solutions:**
1. Check topic remapping: `ros2 topic list`
2. Verify camera is publishing: `ros2 topic hz /camera/image_raw`
3. Lower confidence threshold in params.yaml
4. Check model loaded correctly in logs

#### Issue: RViz shows black screen
**Solutions:**
1. Check topic is correct: `/camera/image_raw`
2. Verify transport hint matches (raw/compressed)
3. Ensure QoS settings match
4. Try different reliability policy

## Performance Optimization Tips

### 1. Model Selection
- Use `yolov8n.pt` for real-time applications
- Use `yolov8s.pt` for balanced performance
- Larger models only for offline processing

### 2. Inference Size
- Default 640x640 is good for most cases
- Reduce to 320x320 for faster inference
- Increase to 1280x1280 for small object detection

### 3. Class Filtering
- Use `target_classes` to detect only needed objects
- Reduces post-processing time

### 4. Depth Processing
- Disable `enable_depth` if not needed
- Saves memory and processing time

### 5. GPU Acceleration
```python
# Automatically detected, but verify:
# In detector.py, check device selection
```

## Advanced Features

### Depth Estimation Integration

When enabled (`enable_depth: true`), the node:
1. Subscribes to depth topic `/camera/depth/image_raw`
2. Extracts depth at detection center point
3. Publishes depth as additional hypothesis in Detection2DArray

View depth values:
```bash
ros2 topic echo /camera/detections
# Look for depth_m class_id in results
```

### Custom Class Filtering

Edit `params.yaml`:
```yaml
target_classes: ["person", "car", "bicycle", "motorcycle"]
```

### Compressed Image Transport

For bandwidth-constrained scenarios:
```yaml
use_compressed: true
```
Input topic becomes `/camera/image_raw/compressed`

## Extending to Mask R-CNN

To extend this pipeline for instance segmentation with Mask R-CNN:

### 1. Install Detectron2
```bash
pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

### 2. Create Mask R-CNN Detector Class

Create `perception_pkg/utils/mask_rcnn_detector.py`:

```python
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np

class MaskRCNNDetector:
    def __init__(self, confidence_threshold=0.5):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.predictor = DefaultPredictor(self.cfg)
    
    def detect(self, image):
        outputs = self.predictor(image)
        # Process outputs to extract masks, boxes, classes
        return outputs
```

### 3. Modify Perception Node

Add parameter to select detector type:
```python
detector_type = self.get_parameter('detector_type').value  # 'yolo' or 'maskrcnn'
```

### 4. Publish Mask Messages

Use `vision_msgs/Detection3DArray` or custom message with mask data.

## Development

### Running Tests

```bash
cd ~/ros2_ws
# Linting
ament_flake8 src/perception_pkg
ament_pep257 src/perception_pkg

# Tests
colcon test --packages-select perception_pkg
```

### Logging Levels

Set via launch argument:
```bash
ros2 run perception_pkg perception_node \
  --ros-args --log-level debug
```

Or in code:
```python
self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
```

## API Reference

### Subscribed Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/image_raw` | `sensor_msgs/Image` | Raw camera feed |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth image (optional) |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Camera calibration |

### Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/camera/detections` | `vision_msgs/Detection2DArray` | Detection results |
| `/camera/detections/visualization` | `sensor_msgs/Image` | Annotated image |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_topic` | string | `/camera/image_raw` | Input image topic |
| `detection_topic` | string | `/camera/detections` | Output detections topic |
| `model_path` | string | `yolov8n.pt` | YOLO model path |
| `confidence_threshold` | float | 0.5 | Minimum detection confidence |
| `inference_size` | int | 640 | Model input size |
| `enable_depth` | bool | true | Enable depth processing |
| `target_classes` | string[] | [] | Filter specific classes |

## License

Apache-2.0

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## Support

For issues and questions:
- ROS 2 Documentation: https://docs.ros.org/en/humble/
- YOLOv8 Documentation: https://docs.ultralytics.com/
- OpenCV Documentation: https://docs.opencv.org/
