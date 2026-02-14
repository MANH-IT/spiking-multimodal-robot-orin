import cv2

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("Không mở được camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Camera started... Nhấn q để thoát")

    while True:
        ret, frame = cap.read()
        print("ret =", ret)

        if not ret:
            print("Không đọc được frame")
            break

        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
