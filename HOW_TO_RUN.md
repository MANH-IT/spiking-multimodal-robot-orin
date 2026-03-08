# 🚀 Hướng dẫn chạy dự án Multi-Modal Robot AI

## 📋 Cài đặt

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. Cài package ở chế độ editable (tùy chọn)
pip install -e .
```

## 🎯 Cách chạy

### **Cách 1: Dùng `main.py` (Khuyến nghị)**

```bash
# Chạy giao diện robot (touchscreen UI)
python main.py ui

# Chạy fullscreen
python main.py ui --fullscreen

# Training model
python main.py train --batch-size 4 --epochs 50 --lr 1e-4

# Đánh giá model
python main.py eval --model path/to/model.pth --dataset path/to/dataset.json

# Demo webcam
python main.py demo

# Test NLP system
python main.py nlp-test
```

### **Cách 2: Dùng script trực tiếp**

```bash
# Giao diện robot
python scripts/run_robot_utt.py
python scripts/run_robot_utt.py --fullscreen

# Demo webcam
python scripts/demo/webcam_demo.py

# Training
python scripts/training/train_with_monitoring.py --batch-size 4 --num-epochs 50

# Evaluation
python scripts/evaluation/comprehensive_eval.py --model model.pth --dataset dataset.json
```

### **Cách 3: Dùng command sau khi cài package**

```bash
# Sau khi cài: pip install -e .
robot-ai ui
robot-ai train --batch-size 4 --epochs 50
robot-ai demo
```

## 📁 Cấu trúc Entry Points

```
main.py                    # Entry point chính (CLI)
├── ui                     # Giao diện touchscreen robot
├── train                  # Training vision model
├── eval                   # Đánh giá model
├── demo                   # Demo webcam
└── nlp-test               # Test NLP system

scripts/
├── run_robot_utt.py       # Script chạy UI (có thể gọi trực tiếp)
├── demo/
│   ├── webcam_demo.py     # Demo webcam
│   └── touchscreen_ui/    # Giao diện touchscreen
├── training/              # Scripts training
└── evaluation/            # Scripts evaluation
```

## 🔧 Cấu hình

### API Keys (tùy chọn)

```bash
# Windows (CMD)
set FPT_AI_KEY=your_key_here

# Windows (PowerShell)
$env:FPT_AI_KEY="your_key_here"

# Linux/Mac
export FPT_AI_KEY=your_key_here
```

Không có API key → hệ thống sẽ dùng mock (vẫn chạy được).

## 📝 Ví dụ chi tiết

### 1. Chạy giao diện robot

```bash
python main.py ui
```

Hoặc:

```bash
python scripts/run_robot_utt.py --fullscreen
```

### 2. Training model

```bash
python main.py train \
    --batch-size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --output-dir experiments/training/my_run
```

### 3. Đánh giá model

```bash
python main.py eval \
    --model experiments/training/my_run/checkpoints/best_model.pth \
    --dataset data/02_processed/vision/coco_format/hilo_annotations_3d.json \
    --export-results
```

### 4. Demo webcam

```bash
python main.py demo
```

### 5. Test NLP

```bash
python main.py nlp-test
```

## ❓ Troubleshooting

**Lỗi import module:**
- Đảm bảo đã cài package: `pip install -e .`
- Hoặc thêm project root vào PYTHONPATH

**Lỗi PyQt5:**
```bash
pip install PyQt5
```

**Lỗi OpenCV:**
```bash
pip install opencv-python
```

**Lỗi API key:**
- Không bắt buộc, hệ thống sẽ dùng mock nếu không có

## 📚 Xem thêm

- `scripts/run_robot_utt.py` - Chi tiết về UI
- `scripts/demo/webcam_demo.py` - Chi tiết về demo
- `scripts/training/` - Chi tiết về training
- `scripts/evaluation/` - Chi tiết về evaluation
