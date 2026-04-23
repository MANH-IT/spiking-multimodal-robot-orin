@echo off
echo ========================================================
echo ROBOT EEEC - DAI HOC GIAO THONG VAN TAI
echo He thong AI Offline (FastAPI + Vision + SNN NLU)
echo ========================================================

echo [1/2] Dang kich hoat moi truong ao...
call venv310\Scripts\activate.bat

echo [2/2] Dang khoi dong AI Server...
start "" http://localhost:8000
python -m uvicorn web_ui.backend.app:app --host 0.0.0.0 --port 8000

pause
