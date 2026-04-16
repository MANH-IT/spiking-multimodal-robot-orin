@echo off
chcp 65001 > nul
cd /d D:\multi_modal_robot_ai

echo ========================================
echo   ROBOT AI - DAI HOC GIAO THONG VAN TAI
echo   Web UI Multi-page
echo ========================================
echo.

echo Khoi dong Backend server...
start "Robot Backend" cmd /k ".venv\Scripts\python.exe web_ui\backend\app.py"

echo Doi server khoi dong...
timeout /t 5 /nobreak > nul

echo Mo trinh duyet...
start http://localhost:8000

echo.
echo Web UI da san sang!
echo    - Trang chu: http://localhost:8000
echo    - Chat: http://localhost:8000/chat
echo    - Ban do: http://localhost:8000/map
echo    - Tin tuc: http://localhost:8000/news
echo.
echo Dong cua so nay de tat server
pause
