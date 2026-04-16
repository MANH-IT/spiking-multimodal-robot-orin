from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initialize YOLOv8 model.
        :param model_path: Path to the YOLOv8 model file.
        """
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.5):
        """
        Detect objects in a frame.
        :param frame: Image frame from camera.
        :param conf: Confidence threshold.
        :return: Results from YOLO.
        """
        results = self.model(frame, conf=conf, verbose=False)
        return results

    def draw_detections(self, frame, results):
        """
        Draw bounding boxes and labels on the frame.
        :param frame: Image frame.
        :param results: YOLO results.
        :return: Annotated frame.
        """
        # results[0].plot() returns the frame with boxes
        return results[0].plot()

if __name__ == "__main__":
    # Test Detector
    detector = YOLODetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.detect(frame)
        annotated_frame = detector.draw_detections(frame, results)
        cv2.imshow("YOLOv8 Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
