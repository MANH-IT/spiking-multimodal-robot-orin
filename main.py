import cv2
from camera.camera import Camera
from detection.yolo_detector import YOLODetector
from tracking.deepsort_tracker import ObjectTracker

def main():
    # 1. Initialize Components
    cam = Camera(0)  # nếu không được thì đổi 1 hoặc 2
    detector = YOLODetector('models/yolov8n.pt')
    tracker = ObjectTracker()

    print("🚀 Starting Robot AI System... Press 'q' to quit.")

    while True:
        # 2. Capture Frame
        ret, frame = cam.get_frame()
        if not ret:
            print("❌ Không lấy được frame")
            break

        # 3. Object Detection (YOLO)
        results = detector.detect(frame)

        # 4. Prepare detections for tracker
        tracker_input = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                tracker_input.append([
                    [int(x1), int(y1), int(w), int(h)],
                    conf,
                    cls
                ])

        # 5. Object Tracking (DeepSORT)
        tracks = tracker.update(frame, tracker_input)

        # 6. Draw Results
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        # 7. Display
        cv2.imshow("Robot AI Vision", frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()