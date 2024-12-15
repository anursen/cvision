from object_detection import ObjectDetection

model_path = 'yolov3.weights'
config_path = 'yolov3.cfg'
classes_path = 'classes.txt'

def main():
    detector = ObjectDetection(model_path, config_path, classes_path)
    detector.detect_objects_from_camera(window_width=800, window_height=600)  # Set desired window size

if __name__ == "__main__":
    main()
