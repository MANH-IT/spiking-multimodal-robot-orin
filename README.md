## Hệ thống nhận diện đối tượng động 3D dùng SNN và camera RGB-D đơn

Đây là repo triển khai cho đề tài:

> **"HỆ THỐNG NHẬN DIỆN ĐỐI TƯỢNG ĐỘNG 3D SỬ DỤNG MẠNG NƠ-RON XUNG VÀ CAMERA RGB-D ĐƠN CHO ROBOT THÔNG MINH: NGHIÊN CỨU VÀ TRIỂN KHAI TRÊN NỀN TẢNG NVIDIA JETSON AGX ORIN" (2026)**

Repo được tổ chức theo hướng **nghiên cứu → prototype → tối ưu Jetson**, với 3 khối chính:

- **Dữ liệu & tiền xử lý** cho HILO + VHOD (sau này).
- **Kiến trúc SNN 3D depth-aware** và baseline ANN/SNN.
- **Triển khai & đánh giá hiệu năng** (PC → Jetson AGX Orin).

---

## 1. Môi trường & phụ thuộc

Python cơ bản:

- `numpy`
- `opencv-python`
- `openni`
- `torch`
- `ultralytics`

Khuyến nghị thêm (cho các bước sau):

- `snntorch` (SNN thực)
- `pyyaml`

Cài đặt nhanh:

```bash
pip install -r requirements.txt
```

---

## 2. Cấu trúc thư mục chính

- `data/`
  - `00_raw/vision/HILO_Dataset/` – dataset HILO (RGB, depth, pose, mesh 3D).
  - `01_interim/vision/` – dữ liệu trung gian:
    - `aligned/{rgb,depth}/` – cặp RGB–Depth đã căn chỉnh.
  - `02_processed/vision/` – dữ liệu đã xử lý:
    - `yolo_format/` – ảnh + nhãn YOLO (hiện là bbox giả để test pipeline).
    - `coco_format/hilo_annotations_3d.json` – annotation 3D skeleton.
- `scripts/data_collection/`
  - `hilo_inspect.py` – xem thống kê HILO.
  - `hilo_to_yolo.py` – khung convert HILO → YOLO (bbox tạm thời).
  - `hilo_make_interim.py` – tạo RGB–Depth aligned.
  - `hilo_make_annotations.py` – skeleton annotation 2D/3D (chưa tính bbox thật).
- `src/common/`
  - `hilo_dataset.py` – loader HILO (scenes, pose, RGB, depth, mask).
  - `annotations_3d.py` – dataclass cho BBox2D/3D, ObjectAnnotation, FrameAnnotation.
- `vision_system/models/snn/`
  - `depth_aware_snn.py` – kiến trúc **DepthAwareSNN** (RGB branch, Depth branch, fusion, temporal, 3D detection head), có tuỳ chọn `use_multiscale`.
  - `multiscale_stdfe.py` – **MultiScaleSTDFE** cho đặc trưng đa tỉ lệ không–thời gian.
  - `memory_optimizer.py` – skeleton quản lý budget bộ nhớ (chuẩn bị cho Jetson).
- `vision_system/data/`
  - `rgbd_sequence_dataset.py` – dataset đọc frame RGB–Depth aligned.
- `vision_system/inference/`
  - `depth_aware_demo.py` – demo inference end-to-end với dữ liệu thật.
  - `benchmark_depth_aware.py` – benchmark latency/FPS cho DepthAwareSNN.
- `vision_system/training/`
  - `depth_aware_train.py` – training loop skeleton cho DepthAwareSNN.

---

## 3. Pipeline dữ liệu HILO

### 3.1 Kiểm tra dataset HILO

```bash
python scripts/data_collection/hilo_inspect.py
```

In ra:

- Số lượng scene.
- Số lượng ảnh (ước lượng từ `camera_poses.json`).
- Thông tin một scene ví dụ.

### 3.2 Tạo RGB–Depth aligned

```bash
python scripts/data_collection/hilo_make_interim.py
```

Kết quả:

- `data/01_interim/vision/aligned/rgb/*.png`
- `data/01_interim/vision/aligned/depth/*.png`

Hiện tại, depth được resize về kích thước RGB; TODO: dùng ma trận calib để warp depth chính xác khi có file intrinsics.

### 3.3 Khung YOLO-format

```bash
python scripts/data_collection/hilo_to_yolo.py
```

Kết quả:

- `data/02_processed/vision/yolo_format/images/...`
- `data/02_processed/vision/yolo_format/labels/...`

Hiện bbox YOLO là **bbox giả che toàn ảnh** (class 0), dùng để test pipeline I/O; sẽ được thay bằng bbox 2D thật khi đã tính được từ pose 3D + intrinsics.

### 3.4 Skeleton annotation 2D/3D

```bash
python scripts/data_collection/hilo_make_annotations.py
```

Kết quả:

- `data/02_processed/vision/coco_format/hilo_annotations_3d.json`

Mỗi frame bao gồm:

- Đường dẫn ảnh RGB/Depth.
- Danh sách đối tượng với `object_id`, `category`, `bbox2d`, `bbox3d` (hiện `bbox2d/bbox3d = None`).

**TODO quan trọng:** khi có intrinsics, bổ sung bước:

- Dùng `camera_poses` + `object_pose_wrt_arc0_image7` + mesh `.obj` để:
  - Tính **bbox 3D** trong hệ toạ độ camera.
  - Project sang **bbox 2D** trên ảnh RGB.

---

## 4. Mô hình DepthAwareSNN & Multi-Scale STDFE

### 4.1 DepthAwareSNN

File: `vision_system/models/snn/depth_aware_snn.py`

- **Input**:
  - `rgb_seq`: `(B, T, 3, H, W)`
  - `depth_seq`: `(B, T, 1, H, W)`
- **Các khối chính**:
  - `SpikingConvLayers` cho RGB và Depth (Conv+BN+ReLU, placeholder SNN).
  - `SpikingAttentionFusion` – attention channel-wise giữa hai nhánh.
  - `SpikingLSTM` – xử lý chuỗi đặc trưng theo thời gian.
  - `SpikingDetection3D` – head dự đoán `(cx, cy, cz, w, h, d)` + logits class.
  - Tuỳ chọn `use_multiscale=True` để dùng `MultiScaleSTDFE` trên chuỗi feature fused.

### 4.2 MultiScaleSTDFE

File: `vision_system/models/snn/multiscale_stdfe.py`

- Scale: `[1, 2, 4]` (tương đương 1x, 1/2x, 1/4x).
- `SpikingTemporalGrad`: gradient theo thời gian `x[t] - x[t-1]`.
- Conv + global pooling trên mỗi scale → concat → `AdaptiveEncoder`.

### 4.3 MemoryOptimizer

File: `vision_system/models/snn/memory_optimizer.py`

- Quản lý budget cho:
  - `rgb_buffer`
  - `depth_buffer`
  - `model_weights`
  - `intermediate`
- Chuẩn bị cho bước tối ưu RAM trên Jetson AGX Orin.

---

## 5. Inference & Benchmark

### 5.1 Demo inference với dữ liệu thật

```bash
python vision_system/inference/depth_aware_demo.py
```

Yêu cầu trước đó đã chạy:

- `scripts/data_collection/hilo_make_interim.py`

Script sẽ:

- Đọc vài frame từ `data/01_interim/vision/aligned/{rgb,depth}`.
- Xây dựng `rgb_seq`, `depth_seq`.
- Chạy `DepthAwareSNN` và in kích thước `bbox_3d` và `logits`.

### 5.2 Benchmark latency & FPS

```bash
python vision_system/inference/benchmark_depth_aware.py
```

Tính:

- Latency trung bình / sequence (ms).
- FPS hiệu dụng = `(B * T) / latency`.

Chạy cả hai cấu hình:

- Baseline (không MultiScaleSTDFE).
- With MultiScaleSTDFE.

Kết quả này là baseline trên máy hiện tại, dùng để so sánh với Jetson AGX Orin sau khi triển khai TensorRT.

---

## 6. Training skeleton

### 6.1 Dataset RGB-D

File: `vision_system/data/rgbd_sequence_dataset.py`

- `RGBDAlignedFrameDataset`:
  - Đọc cặp RGB/Depth từ `data/01_interim/vision/aligned/{rgb,depth}`.
  - Trả về tensors `(3, H, W)` và `(1, H, W)`.

### 6.2 Training loop

File: `vision_system/training/depth_aware_train.py`

Chạy:

```bash
python vision_system/training/depth_aware_train.py
```

Hiện tại:

- T = 1 (frame đơn).
- Loss sử dụng nhãn giả (class 0) để kiểm tra backprop.  
**Cần thay** bằng:

- Nhãn class thật (từ annotation).
- Loss tổng hợp cho bbox 2D/3D + classification.

---

## 7. Roadmap tiếp theo (theo đề tài)

- **ANN → SNN thực**:
  - Thay các module Conv/LSTM/Encoder bằng neuron SNN (SNNTorch).
  - Thiết kế chiến lược huấn luyện (ANN-to-SNN, hybrid, surrogate gradient).
- **Bổ sung intrinsics & bbox 3D/2D thực**:
  - Import file calib camera.
  - Viết hàm:
    - Pose object → bbox 3D camera.
    - Project bbox 3D → bbox 2D trên ảnh.
  - Cập nhật `hilo_make_annotations.py` và `hilo_to_yolo.py`.
- **Triển khai Jetson AGX Orin**:
  - Dockerfile với JetPack 6.0, PyTorch, TensorRT, ROS 2 Humble.
  - Export ONNX → TensorRT, đo latency/power/nhiệt độ.
  - Tích hợp Orbbec SDK, ROS2 node nhận RGB-D và chạy DepthAwareSNN online.

