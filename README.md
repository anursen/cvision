# Object Detection with OpenCV

This project demonstrates object detection using the YOLO model with OpenCV. It can detect objects in images and from a camera feed.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/anursen/cvision.git
    cd cvision
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download YOLO weights and configuration files**:
    - Download the YOLOv3 weights file from [YOLOv3 Weights](https://pjreddie.com/media/files/yolov3.weights)
    - Download the YOLOv3 configuration file from [YOLOv3 Configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg?raw=true)
    - Download the YOLOv2 weights file from [YOLOv2 Weights](https://pjreddie.com/media/files/yolov2.weights)
    - Place these files in the project directory.

    **Note**: The YOLO weights file is large and should not be pushed to the repository. It is included in the `.gitignore` file.

## Usage

1. **Run object detection on an image**:
    ```python
    from object_detection import ObjectDetection

    model_path = 'yolov3.weights'
    config_path = 'yolov3.cfg'
    classes_path = 'classes.txt'

    detector = ObjectDetection(model_path, config_path, classes_path)
    results = detector.detect_objects('path/to/your/image.jpg')
    print(results)
    ```

2. **Run object detection from a camera feed**:
    ```sh
    python run_detection.py
    ```

    Press `q` to quit the camera feed window.

## License

This project is licensed under the MIT License.