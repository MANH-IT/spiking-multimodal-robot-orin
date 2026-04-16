import cv2

class Camera:
    def __init__(self, source=0):
        """
        Initialize the camera.
        :param source: 0 for default webcam, or a path to a video file.
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera source {source}")

    def get_frame(self):
        """
        Capture a frame from the camera.
        :return: Success flag and the frame.
        """
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """
        Release the camera resources.
        """
        self.cap.release()

if __name__ == "__main__":
    # Test camera
    cam = Camera(0)
    while True:
        ret, frame = cam.get_frame()
        if not ret:
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
