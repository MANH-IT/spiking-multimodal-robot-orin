"""
DeepSORT Object Tracker - Robot EEEC
Lightweight tracker cho demo trên Jetson AGX Orin.
Sử dụng thuật toán IOU matching đơn giản thay vì DeepSORT đầy đủ
để giảm dependency và tăng tốc độ xử lý.
"""

import numpy as np


class Track:
    """Đại diện cho một đối tượng đang được theo dõi."""

    _CONFIRMED_HITS = 3  # Số frame liên tiếp để xác nhận track

    def __init__(self, track_id, bbox, cls, conf):
        self.track_id = track_id
        self._bbox = bbox          # [x1, y1, w, h]
        self.cls = cls
        self.conf = conf
        self._hits = 1
        self._age = 0
        self._time_since_update = 0
        self._state = "tentative"  # tentative -> confirmed -> deleted

    def update(self, bbox, cls, conf):
        """Cập nhật track với detection mới."""
        self._bbox = bbox
        self.cls = cls
        self.conf = conf
        self._hits += 1
        self._time_since_update = 0
        if self._hits >= self._CONFIRMED_HITS:
            self._state = "confirmed"

    def mark_missed(self):
        """Đánh dấu track không có detection khớp trong frame này."""
        self._time_since_update += 1
        self._age += 1
        if self._time_since_update > 30:  # Xóa sau 30 frame mất tích
            self._state = "deleted"

    def is_confirmed(self):
        return self._state == "confirmed"

    def is_deleted(self):
        return self._state == "deleted"

    def to_ltrb(self):
        """Trả về bounding box dạng (left, top, right, bottom)."""
        x, y, w, h = self._bbox
        return [x, y, x + w, y + h]


def _iou(bbox1, bbox2):
    """Tính IOU giữa hai bbox dạng [x, y, w, h]."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


class ObjectTracker:
    """
    Lightweight Object Tracker sử dụng IOU matching.
    API tương thích với DeepSORT để dễ nâng cấp sau này.
    """

    def __init__(self, iou_threshold=0.3):
        self.tracks = []
        self._next_id = 1
        self._iou_threshold = iou_threshold

    def update(self, frame, detections):
        """
        Cập nhật tracker với frame và danh sách detections.

        Args:
            frame: numpy array (H, W, 3) - frame hiện tại (dùng cho ReID sau này)
            detections: list of [[x, y, w, h], confidence, class_id]

        Returns:
            list[Track]: Danh sách các track hiện tại
        """
        # Nếu không có detection, đánh dấu tất cả track là missed
        if not detections:
            for track in self.tracks:
                track.mark_missed()
            self.tracks = [t for t in self.tracks if not t.is_deleted()]
            return self.tracks

        det_bboxes = [d[0] for d in detections]
        det_confs = [d[1] for d in detections]
        det_classes = [d[2] for d in detections]

        matched_tracks = set()
        matched_dets = set()

        # Greedy IOU matching
        if self.tracks:
            iou_matrix = np.zeros((len(self.tracks), len(det_bboxes)))
            for i, track in enumerate(self.tracks):
                for j, det_bbox in enumerate(det_bboxes):
                    iou_matrix[i, j] = _iou(track._bbox, det_bbox)

            # Sắp xếp theo IOU giảm dần và match greedy
            while True:
                if iou_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                max_iou = iou_matrix[max_idx]

                if max_iou < self._iou_threshold:
                    break

                t_idx, d_idx = max_idx
                self.tracks[t_idx].update(
                    det_bboxes[d_idx], det_classes[d_idx], det_confs[d_idx]
                )
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

                # Loại bỏ hàng/cột đã match
                iou_matrix[t_idx, :] = 0
                iou_matrix[:, d_idx] = 0

        # Đánh dấu các track không match là missed
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track.mark_missed()

        # Tạo track mới cho các detection không match
        for j in range(len(det_bboxes)):
            if j not in matched_dets:
                new_track = Track(
                    track_id=self._next_id,
                    bbox=det_bboxes[j],
                    cls=det_classes[j],
                    conf=det_confs[j],
                )
                self.tracks.append(new_track)
                self._next_id += 1

        # Xóa các track đã deleted
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        return self.tracks
