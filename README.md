# Object Detection and Tracking with YOLOv4

This project implements real-time object detection and tracking using YOLOv4 and OpenCV. It combines the power of YOLOv4 for object detection with OpenCV's tracking algorithms for efficient real-time tracking.

## Features

- Real-time object detection using YOLOv4
- Object tracking using OpenCV's CSRT tracker
- Support for multiple object tracking
- Automatic re-detection every 30 frames
- Display of object class and confidence score

## Prerequisites

- OpenCV 4.0 or later
- C++ compiler with C++14 support
- CMake 3.10 or later
- CUDA (optional, for GPU acceleration)

## Required Files

Download these files before running the project:
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [yolov4.cfg](https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg)
- [coco.names](https://github.com/AlexeyAB/darknet/blob/master/data/coco.names)

## Project Structure

```
project_folder/
    ├── src/
    │   └── main.cpp
    ├── yolov4.cfg
    ├── coco.names
    ├── CMakeLists.txt
    └── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection-tracking.git
cd object-detection-tracking
```

2. Download the required YOLO files mentioned above and place them in the project root directory.

3. Create a build directory and compile:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Usage

Run the compiled executable:
```bash
./object_detection_tracking
```

Controls:
- Press 'q' to quit the application

## Performance Notes

- The program performs detection every 30 frames and tracking in between
- For better performance, consider using YOLOv4-tiny
- GPU acceleration can be enabled by modifying the backend in the code

## Contributing

Feel free to open issues and pull requests for any improvements.

## License

This project is licensed under the GPLv3 License - see the LICENSE file for details.

## Acknowledgments

- YOLOv4 by Alexey Bochkovskiy
- OpenCV community
