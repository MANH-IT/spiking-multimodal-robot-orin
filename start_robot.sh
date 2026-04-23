#!/bin/bash
echo "========================================================"
echo "ROBOT EEEC - ĐẠI HỌC GTVT"
echo "Hệ thống AI Offline (FastAPI + Vision + SNN NLU)"
echo "========================================================"

echo "[1/2] Đang kích hoạt môi trường..."
source venv310/bin/activate

echo "[2/2] Đang khởi động AI Server..."
python -m uvicorn web_ui.backend.app:app --host 0.0.0.0 --port 8000
