from ultralytics import YOLO

def export_model(model_path='yolov8n.pt'):
    """
    Export YOLOv8 model to ONNX format.
    """
    model = YOLO(model_path)
    # Export to ONNX
    success = model.export(format='onnx', dynamic=True)
    if success:
        print(f"Model exported successfully to {model_path.replace('.pt', '.onnx')}")
    else:
        print("Export failed.")

if __name__ == "__main__":
    export_model()
