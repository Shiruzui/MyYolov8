
# YOLO Object Detection System

This Python-based project enables object detection using YOLOv10 and YOLOv8 models from the ultralytics library, with capabilities for processing videos, webcam feeds, and images.

## Installation

### Prerequisites
- Python 3.8 or < 3.12
- OpenCV
- Numpy
- PyTorch
- ultralytics YOLO models

### Setup

1. **YOLOv10 Installation**
   ```bash
   pip install -q git+https://github.com/THU-MIG/yolov10.git
   ```

2. **YOLOv8 Installation**
   For YOLOv8, use the ultranalytics library:
   ```bash
   pip install ultralytics
   ```

## YOLODetector Class Configuration

### Initialization Parameters
- `yolov8_path` (str): Path to the model file (e.g., 'yolov10x.pt').
- `conf` (float): Confidence threshold for detecting objects. Default is 0.5.
- `filter_labels` (set[str] | None): Set of labels to filter detections. `None` allows all labels.
- `labels` (dict): Optional dictionary mapping label indices to string descriptions.
- `colors` (dict): Optional dictionary specifying colors for each label in RGB.
- `output_dir` (str): Directory where output files will be saved.

### Methods

- **`get_model()`**: Loads the YOLO model based on the provided model path and device availability (GPU/CPU).

- **`update_config(**kwargs)`**: Updates the detector configuration dynamically during runtime. You can pass any configuration parameters (e.g., `conf`, `filter_labels`) as keyword arguments.

- **`show_config()`**: Prints the current configuration settings to the console.

- **`initialize_video_writer(cap, file_name)`**: Initializes a video writer for saving output videos, based on the input capture device and desired output file name.

- **`run_video(video_path, save_output)`**: Processes a video file for object detection. If `save_output` is True, the output is saved using the initialized video writer.

- **`run_webcam(save_output)`**: Processes webcam input for object detection and optionally saves the output.

- **`run_image(image_path, save_output)`**: Processes a single image for object detection and optionally saves the output.

- **`process_stream(cap, save_output, file_name)`**: Handles the continuous input from a video capture device (video file or webcam) and processes each frame through the detector.

- **`save_frame_or_image(frame, save_output, file_name, out=None)`**: Saves a single frame or image, either by writing it to a video or saving it as an image file, depending on the configuration.

- **`process_frame(frame, save_output, file_name, out=None)`**: Processes a single frame for object detection, displays the results, and saves the output if required.

## Additional Features

### Coco Labels and Colors

Utilize the `CocoInfo` class to manage COCO dataset labels and custom color patterns for bounding boxes.

```
coco_info = CocoInfo()
coco_labels = coco_info.get_coco_labels()
coco_colors = coco_info.get_coco_colors_pattern(default=True)
```

## Contributing

Contributions are welcome. Ensure to follow the existing code style and include tests for new or updated functionality.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
