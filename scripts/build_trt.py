from ultralytics import YOLO
import subprocess
import os

def build_tensorrt(model_path='yolov8n.pt'):
    """
    Export YOLOv8 model to TensorRT engine format.
    """
    print(f"Building TensorRT engine for {model_path}...")
    model = YOLO(model_path)
    # ultralytics handles TensorRT export directly if TensorRT is installed
    model.export(format='engine', device=0, half=True)
    print("TensorRT build process finished.")

if __name__ == "__main__":
    build_tensorrt()
