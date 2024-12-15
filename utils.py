import cv2

def draw_boxes(image, detections):
    for detection in detections:
        x, y, w, h = detection["box"]
        label = f"{detection['class']}: {detection['confidence']:.2f}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
