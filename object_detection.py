import cv2
import numpy as np

class ObjectDetection:
    def __init__(self, model_path, config_path, classes_path, use_gpu=False):
        self.net = cv2.dnn.readNet(model_path, config_path)
        if use_gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        with open(classes_path, 'r') as f:
            self.classes = f.read().strip().split('\n')

    def detect_objects(self, image_path):
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        detections = self.net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, w, h) = box.astype("int")
                    x = int(centerX - (w / 2))
                    y = int(centerY - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        result = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                result.append({
                    "class": self.classes[class_ids[i]],
                    "confidence": confidences[i],
                    "box": box
                })

        return result

    def detect_objects_from_camera(self, camera_index=0, window_width=800, window_height=600):
        cap = cv2.VideoCapture(camera_index)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            detections = self.net.forward(output_layers)

            boxes = []
            confidences = []
            class_ids = []

            for output in detections:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        box = detection[0:4] * np.array([width, height, width, height])
                        (centerX, centerY, w, h) = box.astype("int")
                        x = int(centerX - (w / 2))
                        y = int(centerY - (h / 2))
                        boxes.append([x, y, int(w), int(h)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            result = []
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    result.append({
                        "class": self.classes[class_ids[i]],
                        "confidence": confidences[i],
                        "box": box
                    })

            from utils import draw_boxes  # Import draw_boxes here to avoid circular import
            frame = draw_boxes(frame, result)
            frame = cv2.resize(frame, (window_width, window_height))  # Resize the frame
            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
