@echo off
REM Script cài đặt dependencies cho Windows
REM Sử dụng python -m pip để tránh lỗi với đường dẫn có khoảng trắng

echo ========================================
echo   Cài đặt Dependencies
echo ========================================
echo.

REM Kiểm tra Python
python --version
if errorlevel 1 (
    echo ERROR: Python không tìm thấy!
    pause
    exit /b 1
)

echo.
echo [1/3] Cài đặt datasets và huggingface-hub...
python -m pip install datasets huggingface-hub
if errorlevel 1 (
    echo ERROR: Không thể cài datasets và huggingface-hub
    pause
    exit /b 1
)

echo.
echo [2/3] Cài đặt từ requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Một số packages có thể chưa được cài đặt
)

echo.
echo [3/3] Cài đặt package ở chế độ editable...
python -m pip install -e .
if errorlevel 1 (
    echo WARNING: Không thể cài ở chế độ editable
)

echo.
echo ========================================
echo   Hoàn tất!
echo ========================================
echo.
echo Test import:
python -c "import datasets; print('  OK: datasets')"
python -c "import huggingface_hub; print('  OK: huggingface_hub')"
echo.

pause
