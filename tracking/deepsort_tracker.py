from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

class ObjectTracker:
    def __init__(self, max_age=30):
        """
        Initialize DeepSORT tracker.
        """
        self.tracker = DeepSort(max_age=max_age)

    def update(self, frame, detections):
        """
        Update tracker with new detections.
        detections format: [[x1, y1, w, h], confidence, class_id]
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        return tracks

    def draw_tracks(self, frame, tracks):
        """
        Draw tracking IDs and boxes.
        """
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb() # Left, Top, Right, Bottom
            
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame
