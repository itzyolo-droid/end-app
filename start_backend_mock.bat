@echo off
echo ===================================================
echo Starting Backend Server in MOCK Mode
echo ===================================================
echo.

if not exist venv (
    echo [1/3] Creating Python virtual environment venv...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create venv. Please make sure Python is installed and added to PATH.
        pause
        exit /b 1
    )
    echo.
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate
echo.

echo [3/3] Installing lightweight mock dependencies...
python -m pip install --upgrade pip
pip install -r backend/requirements-dev.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies.
    pause
    exit /b 1
)
echo.

cd backend
echo Starting FastAPI server on http://127.0.0.1:8000
echo.
python main.py
pause
